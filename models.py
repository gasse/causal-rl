import typing
import torch


class AugmentedModel(torch.nn.Module):

    """ Augmented Model Base Class """

    def log_q_s(self, s=None):
        raise NotImplementedError

    def log_q_snext_sa(self, a, s=None, snext=None):
        raise NotImplementedError

    def log_q_o_s(self, o, s=None):
        raise NotImplementedError

    def log_q_r_s(self, r, s=None):
        raise NotImplementedError

    def log_q_d_s(self, d, s=None):
        raise NotImplementedError

    def log_q_a_s(self, a, s=None):
        raise NotImplementedError

    def log_q_s_h(self, regime, loq_q_sprev_hprev, a, o, r, d, with_done=False):

        assert (loq_q_sprev_hprev is None) == (a is None)

        # hprev = (o_0, r_0, d_0, a_0, ..., o_t-1, r_t-1, d_t-1)
        # sprev = s_t-1
        # a = a_t-1
        # o = o_t
        # r = r_t
        # d = d_t
        # h =     (o_0, r_0, d_0, a_0, ..., o_t-1, r_t-1, d_t-1, a_t-1, o_t, r_t, d_t)
        # s = s_t

        no_hprev = (a is None)

        if no_hprev:
            # (batch_size, s_nvals)
            log_q_s_hpreva = self.log_q_s()

        else:

            # (batch_size, s_nvals) - (batch, sprev)
            log_q_a_sprev = self.log_q_a_s(a=a)
            log_q_a_sprev = log_q_a_sprev * (1 - d).unsqueeze(-1)  # discard actions if done=True
            log_q_a_sprev = log_q_a_sprev * (1 - regime).unsqueeze(-1)  # discard actions in interventional regime

            # (batch_size, s_nvals) - (batch, sprev)
            log_q_spreva_hprev = loq_q_sprev_hprev + log_q_a_sprev

            # (batch_size,) - (batch,)
            log_q_a_hprev = torch.logsumexp(log_q_spreva_hprev, dim=-1)

            # (batch_size, s_nvals) - (batch, sprev)
            log_q_sprev_hpreva = log_q_spreva_hprev - log_q_a_hprev.unsqueeze(-1)

            # (batch_size, s_nvals, s_nvals) - (batch, sprev, s)
            log_q_s_spreva = self.log_q_snext_sa(a=a)

            # (batch_size, s_nvals, s_nvals) - (batch, sprev, s)
            loq_q_sprevs_hpreva = log_q_sprev_hpreva.unsqueeze(-1) + log_q_s_spreva

            # (batch_size, s_nvals) - (batch, s)
            log_q_s_hpreva = torch.logsumexp(loq_q_sprevs_hpreva, dim=-2)

        log_q_o_s = self.log_q_o_s(o=o)
        log_q_r_s = self.log_q_r_s(r=r)
        log_q_d_s = self.log_q_d_s(d=d) if with_done else 0

        # (batch_size, s_nvals)
        log_q_ord_s = log_q_o_s + log_q_r_s + log_q_d_s

        # (batch_size, s_nvals)
        log_q_sord_hpreva = log_q_s_hpreva + log_q_ord_s

        # (batch_size,)
        log_q_ord_hpreva = torch.logsumexp(log_q_sord_hpreva, dim=-1)

        # (batch_size, s_nvals)
        log_q_s_h = log_q_sord_hpreva - log_q_ord_hpreva.unsqueeze(-1)

        return log_q_s_h

    @torch.jit.export
    def log_prob_joint(self, regime, episode, states, with_done=False):
        log_prob = 0

        n_transitions = len(episode) // 4
        for t in range(n_transitions + 1):

            # s_t, o_t, r_t, d_t
            state, obs, reward, done = states[t], episode[4*t], episode[4*t+1], episode[4*t+2]

            if t == 0:
                was_done = torch.zeros_like(done)

                # (batch_size, )
                log_q_s_saprev = self.log_q_s(s=state)

            else:

                # safety fix, in case a done flag goes back down
                done = torch.max(was_done, done)

                # s_t-1, a_t-1
                state_prev, action_prev = states[t-1], episode[4*t-1]

                # (batch_size, )
                log_q_s_saprev = self.log_q_snext_sa(a=action_prev, s=state_prev, snext=state)

            # (batch_size, )
            log_q_o_s = self.log_q_o_s(o=obs, s=state)
            log_q_r_s = self.log_q_r_s(r=reward, s=state)

            if with_done:
                # (batch_size, )
                log_q_d_s = self.log_q_d_s(d=done, s=state)
            else:
                log_q_d_s = 0

            # a_t (if any)
            if t < n_transitions:
                action = episode[4*(t+1)-1]

                # (batch_size, )
                log_q_a_s = self.log_q_a_s(a=action, s=state)
                log_q_a_s = log_q_a_s * (1 - done)  # discard actions if done=True
                log_q_a_s = log_q_a_s * (1 - regime)  # discard actions in interventional regime
            else:
                log_q_a_s = 0

            # (batch_size, )
            log_q_sorda_saprev = log_q_s_saprev + log_q_o_s + log_q_r_s + log_q_d_s + log_q_a_s

            # discard transitions after done=True (due to padding)
            log_q_sorda_saprev = log_q_sorda_saprev * (1 - was_done)

            # (batch_size, )
            log_prob = log_prob + log_q_sorda_saprev

            was_done = done

        return log_prob

    @torch.jit.export
    def log_prob(self, regime: torch.Tensor, episode: typing.List[torch.Tensor], with_done: bool=False, return_loq_q_s_h: bool=False):

        # if requested, store all q(s_t | h_t) and q(s_t+1 | h_t) during forward
        if return_loq_q_s_h:
            seq_loq_q_s_h = []

        log_prob = 0
        done = torch.tensor([0.])

        n_transitions = len(episode) // 4
        for t in range(n_transitions + 1):

            # o_t, r_t, d_t
            obs, reward, done = episode[4*t], episode[4*t+1], episode[4*t+2]

            if t == 0:
                was_done = torch.zeros_like(done)

                # (batch_size, s_nvals)
                log_q_s_hprev = self.log_q_s().unsqueeze(0)

            else:
                # safety fix, in case a done flag goes back down
                done = torch.max(was_done, done)

                # (batch_size, s_nvals)
                log_q_s_hprev = log_q_snext_h

            # (batch_size, s_nvals)
            log_q_o_s = self.log_q_o_s(o=obs)
            log_q_r_s = self.log_q_r_s(r=reward)

            if with_done:
                # (batch_size, s_nvals)
                log_q_d_s = self.log_q_d_s(d=done)
            else:
                log_q_d_s = 0

            # a_t (if any)
            if t < n_transitions:
                action = episode[4*(t+1)-1]

                # (batch_size, s_nvals)
                log_q_a_s = self.log_q_a_s(a=action)
                log_q_a_s = torch.where((done==1).unsqueeze(-1), torch.zeros_like(log_q_a_s), log_q_a_s)  # discard actions if done=True
                log_q_a_s = torch.where((regime==1).unsqueeze(-1), torch.zeros_like(log_q_a_s), log_q_a_s)  # discard actions in interventional regime
            else:
                log_q_a_s = 0

            # hprev = (o_0, r_0, d_0, a_0, ..., o_t-1, r_t-1, d_t-1, a_t-1)

            # (batch_size, s_nvals)
            log_q_sorda_hprev = log_q_s_hprev + log_q_o_s + log_q_r_s + log_q_d_s + log_q_a_s

            # (batch_size, )
            log_q_orda_hprev = torch.logsumexp(log_q_sorda_hprev, dim=-1)

            # discard transitions after done=True (due to padding)
            log_q_orda_hprev = log_q_orda_hprev * (1 - was_done)

            if t == 0:
                # (batch_size, )
                log_prob = log_prob + log_q_orda_hprev
            else:
                # (batch_size, )
                log_prob = torch.where(log_prob.isinf(), log_prob, log_prob + log_q_orda_hprev)  # bugfix, otherwise NaNs will appear

            # h = (o_0, r_0, d_0, a_0, ..., o_t, r_t, d_t, a_t)

            # (batch_size, s_nvals)
            log_q_s_h = log_q_sorda_hprev - log_q_orda_hprev.unsqueeze(1)

            # snext = s_t+1

            if t < n_transitions:
                # (batch_size, s_nvals, s_nvals)
                log_q_ssnext_h = log_q_s_h.unsqueeze(2) + self.log_q_snext_sa(a=action)
                # (batch_size, s_nvals)
                log_q_snext_h = torch.logsumexp(log_q_ssnext_h, dim=1)

            else:
                log_q_snext_h = None

            # if requested, store all q(s_t | h_t) and q(s_t+1 | h_t) during forward
            if return_loq_q_s_h:
                seq_loq_q_s_h.append((log_q_s_h, log_q_snext_h))

            was_done = done

        if return_loq_q_s_h:
            return log_prob, seq_loq_q_s_h
        else:
            return log_prob

    @torch.jit.export
    def sample_states(self, regime: torch.Tensor, episode: typing.List[torch.Tensor], with_done: bool=False):

        with torch.no_grad():

            # collect all q(s_t | h_t) with a forward pass
            _, seq_log_q_s_h = self.log_prob(regime, episode, with_done=with_done, return_loq_q_s_h=True)

            # collect all s_t ~ q(s_t | h_t, s_t+1) with a backward pass
            states = []
            n_transitions = len(episode) // 4
            for t in range(n_transitions, -1, -1):
                log_q_s_h, log_q_snext_h = seq_log_q_s_h[t]

                if t == n_transitions:
                    # (batch_size, s_nvals)
                    log_q_s_hsnext = log_q_s_h

                else:
                    action = episode[4*(t+1)-1]

                    # (batch_size, s_nvals)
                    log_q_snext_sa = self.log_q_snext_sa(a=action, snext=state)

                    # (batch_size, s_nvals)
                    log_q_ssnext_h = log_q_s_h + log_q_snext_sa

                    # (batch_size, s_nvals)
                    log_q_s_hsnext = log_q_ssnext_h - log_q_snext_h

                state = torch.distributions.one_hot_categorical.OneHotCategorical(
                    logits=log_q_s_hsnext,
                ).sample()

                states.insert(0, state)

        return states

    @torch.jit.export
    def loss_nll(self, regime: torch.Tensor, episode: typing.List[torch.Tensor], with_done: bool=False):
        return -self.log_prob(regime, episode, with_done=with_done)

    @torch.jit.export
    def loss_em(self, regime: torch.Tensor, episode: typing.List[torch.Tensor], with_done: bool=False):
        states = self.sample_states(regime, episode, with_done=with_done)
        return -self.log_prob_joint(regime, episode, states, with_done=with_done)


class TabularAugmentedModel(AugmentedModel):
    
    """ Learnable Augmented Model using tabular probability distribution parameters. """

    def __init__(self, s_nvals, o_nvals, a_nvals, r_nvals):
        super().__init__()
        self.s_nvals = s_nvals
        self.o_nvals = o_nvals
        self.a_nvals = a_nvals
        self.r_nvals = r_nvals

        # p(s_0)
        self.params_s = torch.nn.Parameter(torch.empty([s_nvals]))
        torch.nn.init.normal_(self.params_s)

        # p(s_t+1 | s_t, a_t)
        self.params_s_sa = torch.nn.Parameter(torch.empty([s_nvals, a_nvals, s_nvals]))
        torch.nn.init.normal_(self.params_s_sa)

        # p(o_t | s_t)
        self.params_o_s = torch.nn.Parameter(torch.empty([s_nvals, o_nvals]))
        torch.nn.init.normal_(self.params_o_s)

        # p(r_t | s_t)
        self.params_r_s = torch.nn.Parameter(torch.empty([s_nvals, r_nvals]))
        torch.nn.init.normal_(self.params_r_s)

        # p(d_t | s_t)
        self.params_d_s = torch.nn.Parameter(torch.empty([s_nvals]))
        torch.nn.init.normal_(self.params_d_s)

        # p(a_t | s_t, i=0)
        self.params_a_s = torch.nn.Parameter(torch.empty([s_nvals, a_nvals]))
        torch.nn.init.normal_(self.params_a_s)

    # @torch.jit.export
    @torch.jit.ignore
    def log_q_s(self, s: typing.Optional[torch.Tensor]=None):

        """ Log proba of state distribution p(s) """ 

        log_q_s = torch.nn.functional.log_softmax(self.params_s, dim=-1)

        if s is not None:
            assert s.ndim == 1  # batched
            log_q_s = log_q_s[s]

        return log_q_s

    # @torch.jit.export
    @torch.jit.ignore
    def log_q_snext_sa(self, a: torch.Tensor,
                       s: typing.Optional[torch.Tensor]=None,
                       snext: typing.Optional[torch.Tensor]=None):

        """ Log proba of state transition distribution p(s|s, a). """ 

        # (s_nvals, a_nvals, s_nvals)
        log_q_snext_sa = torch.nn.functional.log_softmax(self.params_s_sa, dim=-1)
        indices = []

        if s is not None:
            assert s.ndim == 1  # batched
            indices.insert(0, s)

        assert a.ndim == 1  # batched
        indices.insert(0, a)
        log_q_snext_sa = log_q_snext_sa.permute(1, 0, 2)

        if snext is not None:
            assert snext.ndim == 1  # batched
            indices.insert(0, snext)
            log_q_snext_sa = log_q_snext_sa.permute(2, 0, 1)

        log_q_snext_sa = log_q_snext_sa[indices]

        return log_q_snext_sa

    # @torch.jit.export
    @torch.jit.ignore
    def log_q_o_s(self, o: torch.Tensor,
                  s: typing.Optional[torch.Tensor]=None):

        """ Log proba of conditional observation distribution from state p(o|s). """ 

        log_q_o_s = torch.nn.functional.log_softmax(self.params_o_s, dim=-1)

        indices = []

        if s is not None:
            assert s.ndim == 1  # batched
            indices.insert(0, s)

        assert o.ndim == 1  # batched
        indices.insert(0, o)
        log_q_o_s = log_q_o_s.permute(1, 0)

        log_q_o_s = log_q_o_s[indices]

        return log_q_o_s

    # @torch.jit.export
    @torch.jit.ignore
    def log_q_a_s(self, a: torch.Tensor,
                  s: typing.Optional[torch.Tensor]=None):

        """ Log proba of conditional action distribution from state p(a|s). """ 

        log_q_a_s = torch.nn.functional.log_softmax(self.params_a_s, dim=-1)

        indices = []

        if s is not None:
            assert s.ndim == 1  # batched
            indices.insert(0, s)

        assert a.ndim == 1  # batched
        indices.insert(0, a)
        log_q_a_s = log_q_a_s.permute(1, 0)

        log_q_a_s = log_q_a_s[indices]

        return log_q_a_s

    # @torch.jit.export
    @torch.jit.ignore
    def log_q_r_s(self, r: torch.Tensor,
                  s: typing.Optional[torch.Tensor]=None):

        """ Log proba of conditional reward distribution from state p(r|s). """ 

        log_q_r_s = torch.nn.functional.log_softmax(self.params_r_s, dim=-1)

        indices = []

        if s is not None:
            assert s.ndim == 1  # batched
            indices.insert(0, s)

        indices.insert(0, r)
        log_q_r_s = log_q_r_s.permute(1, 0)

        log_q_r_s = log_q_r_s[indices]

        return log_q_r_s

    # @torch.jit.export
    @torch.jit.ignore
    def log_q_d_s(self, d: torch.Tensor,
                  s: typing.Optional[torch.Tensor]=None):

        """ Log proba of conditional flagDone distribution from state p(d|s). """
        assert d.ndim == 1  # batched

        if s is not None:
            assert s.ndim == 1  # batched
            params_d_s = self.params_d_s[s]

        else:
            d, params_d_s = torch.broadcast_tensors(
                d.unsqueeze(1),
                self.params_d_s.unsqueeze(0))
    
        log_q_d_s = torch.distributions.bernoulli.Bernoulli(
            logits=params_d_s).log_prob(d.to(dtype=torch.float))

        return log_q_d_s

    def get_settings(self):

        """ Return a dictionnary with all proba distributions. 
        Straightforward, as we provided previously those same proba distributions. """
        
        settings = {}
        settings["p_s"] = torch.nn.functional.softmax(self.params_s, dim=-1)
        settings["p_o_s"] = torch.nn.functional.softmax(self.params_o_s, dim=-1)
        settings["p_s_sa"] = torch.nn.functional.softmax(self.params_s_sa, dim=-1)
        settings["p_r_s"] = torch.nn.functional.softmax(self.params_r_s, dim=-1)

        q_d_s = torch.sigmoid(self.params_d_s)
        settings["p_d_s"] = torch.stack([1-q_d_s, q_d_s], dim=-1)
        
        settings["p_or_s"] = settings["p_r_s"].unsqueeze(-2) * settings["p_o_s"].unsqueeze(-1)
        
        return settings

    def set_probs(self, p_s=None, p_o_s=None, p_s_sa=None, p_r_s=None, p_d_s=None, p_a_s=None):
        with torch.no_grad():
            if p_s is not None:
                self.params_s[:] = torch.as_tensor(p_s).log()
            if p_o_s is not None:
                self.params_o_s[:] = torch.as_tensor(p_o_s).log()
            if p_s_sa is not None:
                self.params_s_sa[:] = torch.as_tensor(p_s_sa).log()
            if p_r_s is not None:
                self.params_r_s[:] = torch.as_tensor(p_r_s).log()
            if p_d_s is not None:
                self.params_d_s[:] = torch.as_tensor(p_d_s).log()
            if p_a_s is not None:
                self.params_a_s[:] = torch.as_tensor(p_a_s).log()
