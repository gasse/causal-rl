import torch
import datetime
import numpy as np


def print_log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

def plot_mean_std(ax, x, y, label, color):
    ax.plot(x, y.mean(0), label=label, color=color)
    ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=color, alpha=0.2)

def plot_mean_lowhigh(ax, x, mean, low, high, label, color):
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, low, high, color=color, alpha=0.2)

def compute_central_tendency_and_error(id_central, id_error, sample):
    if id_central == 'mean':
        central = np.nanmean(sample, axis=0)
    elif id_central == 'median':
        central = np.nanmedian(sample, axis=0)
    else:
        raise NotImplementedError

    if isinstance(id_error, int):
        low = np.nanpercentile(sample, q=int((100 - id_error) / 2), axis=0)
        high = np.nanpercentile(sample, q=int(100 - (100 - id_error) / 2), axis=0)
    elif id_error == 'std':
        low = central - np.nanstd(sample, axis=0)
        high = central + np.nanstd(sample, axis=0)
    elif id_error == 'sem':
        low = central - np.nanstd(sample, axis=0) / np.sqrt(sample.shape[0])
        high = central + np.nanstd(sample, axis=0) / np.sqrt(sample.shape[0])
    else:
        raise NotImplementedError

    return central, low, high

#################################### Metrics ####################################

@torch.jit.script
def kl_div(p, q, ndims: int=1):
#     div = torch.nn.functional.kl_div(p, q, reduction='none')
    div = p * (torch.log(p) - torch.log(q))
    div[p == 0] = 0  # NaNs quick fix
    dims = [i for i in range(-1, -(ndims+1), -1)]
    div = div.sum(dims)
    return div

@torch.jit.script
def js_div(p, q, ndims: int=1):
    m = (p + q) / 2
    div = (kl_div(p, m, ndims) + kl_div(q, m, ndims)) / 2
    return div

#############################################################################

def construct_dataset(env, policy, n_samples, privileged):

    """ Construct a dataset (of n samples) by collecting rollouts using a given 
        policy in a given environment """

    if privileged:
        regime = torch.tensor(0)
    else:
        regime = torch.tensor(1)

    data = []
    for _ in range(n_samples):
        episode = []

        policy.reset()
        obs, reward, done, info = env.reset()
        episode += [obs, reward, done]

        while not done:
            if privileged:
                state = info["state"]
                action = policy.action(state)
            else:
                action = policy.action(obs, reward)

            obs, reward, done, info = env.step(action)
            episode += [action, obs, reward, done]

        data.append((regime, episode))
    return data

#################################################################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

####################################### Empirical JS #######################################


def cross_entropy_empirical(model_q, data_p, batch_size, with_done=False):

    device = next(model_q.parameters()).device

    dataloader_p = torch.utils.data.DataLoader(Dataset(data_p), batch_size=batch_size)

    ce = 0

    for batch in dataloader_p:
        regime, episode = batch
        regime, episode = regime.to(device), [tensor.to(device) for tensor in episode]

        log_prob_q = model_q.log_prob(regime, episode, with_done=with_done)

        ce += -log_prob_q.sum(dim=0)

    ce /= len(data_p)

    return ce


def kl_div_empirical(model_p, model_q, data_p, batch_size, with_done=False):

    assert next(model_q.parameters()).device == next(model_p.parameters()).device

    device = next(model_p.parameters()).device

    # Build DataLoaders
    dataloader_p = torch.utils.data.DataLoader(Dataset(data_p), batch_size=batch_size)

    # KL(p|q) = E x~p(x) [log(p(x)) - log(q(x))]
    kl_p_q = 0

    for batch in dataloader_p:
        regime, episode = batch
        regime, episode = regime.to(device), [tensor.to(device) for tensor in episode]

        log_prob_q = model_q.log_prob(regime, episode, with_done=with_done)
        log_prob_p = model_p.log_prob(regime, episode, with_done=with_done)

        kl_p_q += (log_prob_p - log_prob_q).sum(dim=0)

    kl_p_q /= len(data_p)

    return kl_p_q


def js_div_empirical(model_q, model_p, data_q, data_p, batch_size, with_done=False):

    assert next(model_q.parameters()).device == next(model_p.parameters()).device

    device = next(model_p.parameters()).device

    # Build DataLoaders
    dataloader_q = torch.utils.data.DataLoader(Dataset(data_q), batch_size=batch_size)
    dataloader_p = torch.utils.data.DataLoader(Dataset(data_p), batch_size=batch_size)

    # m = (p + q) / 2

    # KL(p|m) = E x~p(x) [log(p(x)) - log(q(x) + p(x)) + log(2)]
    kl_p_m = 0

    for batch in dataloader_p:
        regime, episode = batch
        regime, episode = regime.to(device), [tensor.to(device) for tensor in episode]

        log_prob_q = model_q.log_prob(regime, episode, with_done=with_done)
        log_prob_p = model_p.log_prob(regime, episode, with_done=with_done)
        log_prob_m = torch.logsumexp(torch.stack([log_prob_q, log_prob_p], dim=0), dim=0)  # - torch.log(torch.tensor(2, device=device))

        kl_p_m += (log_prob_p - log_prob_m).sum(dim=0)

    kl_p_m /= len(data_p)
    kl_p_m += torch.log(torch.tensor(2, device=device))

    # KL(q|m) = E x~q(x) [log(q(x)) - log(q(x) + p(x)) + log(2)]
    kl_q_m = 0

    for batch in dataloader_q:
        regime, episode = batch
        regime, episode = regime.to(device), [tensor.to(device) for tensor in episode]

        log_prob_q = model_q.log_prob(regime, episode, with_done=with_done)
        log_prob_p = model_p.log_prob(regime, episode, with_done=with_done)
        log_prob_m = torch.logsumexp(torch.stack([log_prob_q, log_prob_p], dim=0), dim=0)  # - torch.log(torch.tensor(2, device=device))

        kl_q_m += (log_prob_q - log_prob_m).sum(dim=0)

    kl_q_m /= len(data_q)
    kl_q_m += torch.log(torch.tensor(2, device=device))

    # JS(p|q) = (KL(p|m) + KL(q|m)) / 2

    return (kl_q_m + kl_p_m) / 2
