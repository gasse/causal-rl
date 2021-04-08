import torch


class TabularPOMDP():

    def __init__(self, p_s, p_or_s, p_s_sa, episode_length):
        """ Tabular POMDP environment.
            p_s: p(s) initial distribution of the latent state.
            p_or_s: p(or|s) distribution of observation and reward given latent state.
            p_s_sa: p(s|s,a) transition distribution of next latent state given current state and action.
            episode_length: length of an episode.
        """

        # POMDP dynamics
        self.p_s = torch.as_tensor(p_s)  # size (s_nvals,)
        self.p_s_sa = torch.as_tensor(p_s_sa)  # size (s_nvals, a_nvals, s_nvals)
        self.p_or_s = torch.as_tensor(p_or_s)  # size (s_nvals, o_nvals, r_nvals)
        self.episode_length = episode_length

        # sanity checks
        assert self.p_s.ndim == 1
        assert self.p_s_sa.ndim == 3
        assert self.p_or_s.ndim == 3
        assert self.p_s.shape[0] == self.p_s_sa.shape[0]
        assert self.p_s.shape[0] == self.p_s_sa.shape[2]
        assert self.p_s.shape[0] == self.p_or_s.shape[0]

        self.s_nvals = self.p_s.shape[0]
        self.a_nvals = self.p_s_sa.shape[1]
        self.o_nvals = self.p_or_s.shape[1]
        self.r_nvals = self.p_or_s.shape[2]

        # Initialize current time step
        self.current_step = -1

    def reset(self):

        self.current_step = 0

        # Reset the environment to an initial state sampled from p(s)
        self.s =  torch.distributions.categorical.Categorical(probs=self.p_s,).sample()

        # Sample o, r from p(o,r|s)
        o, r, done, info = self.sample_ordi_s()

        return o, r, done, info

    def step(self, action):

        """ Take a step in the environment
            1. Generate new hidden state p(s|s,a)
            2. Get Obeservation, Reward from p(o,r|s)
            3. Update done flag
            3. Return obs, reward, done, info
        """

        self.current_step += 1

        # Sample next hidden state from current state and action p(s|s,a)
        self.s = torch.distributions.categorical.Categorical(probs=self.p_s_sa[self.s, action],).sample()

        # Sample reward and observation from current hidden state p(o,r|s)
        o, r, done, info = self.sample_ordi_s()

        return o, r, done, info

    def sample_ordi_s(self):

        # Sample from joint distribution
        _or = torch.distributions.categorical.Categorical(probs=self.p_or_s[self.s].view(-1),).sample()
        o = _or // self.p_or_s.shape[2]
        r = _or % self.p_or_s.shape[2]

        done = torch.tensor(self.current_step >= self.episode_length, dtype=torch.long)
        info = {"state": self.s}

        return o, r, done, info


class RewardMapWrapper():

    def __init__(self, env, reward_map):
        self.env = env
        self.reward_map = reward_map

    def reset(self):
        o, r, d, info = self.env.reset()

        # return mapped reward
        return o, self.reward_map[r], d, info

    def step(self, action):
        o, r, d, info = self.env.step(action)

        # return mapped reward
        return o, self.reward_map[r], d, info


class BeliefStateWrapper():

    def __init__(self, env, belief_model, with_done=False):
        self.env = env
        self.belief_model = belief_model
        self.with_done = with_done

    def update_belief_state(self, a, o, r, d):

        ''' Update the proba distribution of hidden state representation p(s|h).
        Ie the distribution of the hidden state estimated s given the whole history '''

        with torch.no_grad():
            self.belief_state = self.belief_model.log_q_s_h(
                regime=torch.tensor(1.).unsqueeze(0),  # interventional regime
                loq_q_sprev_hprev=None if self.belief_state is None else self.belief_state.unsqueeze(0), 
                a=None if a is None else torch.as_tensor(a).unsqueeze(0),
                o=o.unsqueeze(0),
                r=r.unsqueeze(0),
                d=d.unsqueeze(0), 
                with_done=self.with_done).squeeze(0)

    def reset(self):

        ''' Reset hidden state belief  '''

        o, r, d, info = self.env.reset()

        self.belief_state = None
        self.update_belief_state(None, o, r, d)

        # return belief state as the observation
        return torch.exp(self.belief_state), r, d, info

    def step(self, action):

        ''' Take a step, update hidden state beliefs and
        return the hidden state p(s|h) as observation '''

        o, r, d, info = self.env.step(action)

        self.update_belief_state(action, o, r, d)

        # return belief state as the observation
        return torch.exp(self.belief_state), r, d, info
