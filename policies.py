import torch

class Policy(torch.nn.Module):

    """ Empty class, must include
        1. a Reset method
        3. a Action method to act in environment from 4-uplets (o, r, d, info)
    """ 

    def reset(self):
        raise NotImplemented

    def action(self, obs, reward):
        raise NotImplemented

class UniformPolicy(Policy):

    """ Uniform policy to act within the environement with random distributed actions """

    def __init__(self, a_nvals):
        super().__init__()

        self.a_nvals = a_nvals

    def reset(self):
        pass

    def action(self, obs, reward):

        action = torch.distributions.categorical.Categorical(
            probs=torch.ones(self.a_nvals)/self.a_nvals).sample()
 
        return action

class PrivilegedPolicy(Policy):

    """ Expert Policy Class that chooses its actions from the hidden state s """

    def __init__(self, p_a_s):
        super().__init__()
        self.probs_a_s = torch.as_tensor(p_a_s)  # p(a_t | s_t, i=0)

    def reset(self):
        pass

    def action(self, state):

        action = torch.distributions.categorical.Categorical(probs=self.probs_a_s[state]).sample()

        return action
