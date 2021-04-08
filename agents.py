import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from utils import print_log


class ActorCritic(nn.Module):
    
    def __init__(self, s_nvals, a_nvals, hidden_size=32):

        super(ActorCritic, self).__init__()

        self.a_nvals = a_nvals
        self.s_nvals = s_nvals
        self.hidden_size = hidden_size

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(self.s_nvals, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.a_nvals),
            torch.nn.LogSoftmax(dim=-1),
        )

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(self.s_nvals, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def get_action(self, state, with_log_prob=False, greedy=False):
        policy_log_probs = self.actor(state)

        if greedy:
            action = policy_log_probs.argmax(-1)
        else:
            action = torch.distributions.categorical.Categorical(logits=policy_log_probs).sample()
 
        if with_log_prob:
            log_prob = policy_log_probs[..., action]
            return action, log_prob
        else:
            return action

    def get_value(self, state):
        return self.critic(state)


def run_episode(env, agent, max_env_steps):

    action_log_probs, rewards, values = [], [], []

    with torch.no_grad():
        state, reward, done, _ = env.reset()
        t = 0

    while not done:

        action, action_log_prob = agent.get_action(state, with_log_prob=True)
        value = agent.get_value(state)

        with torch.no_grad():
            state, reward, done, _ = env.step(action)
            t += 1

        action_log_probs.append(action_log_prob)
        values.append(value)
        rewards.append(reward)

        if t >= max_env_steps:
            break

    return action_log_probs, values, rewards


def loss_episode(env, agent, gamma, max_steps_per_episode):

    action_log_probs, values, rewards = run_episode(env, agent, max_steps_per_episode)
 
    # compute (discounted) returns from rewards
    returns = []
    current_return = 0.
    for reward in reversed(rewards):
        current_return = reward + gamma * current_return
        returns.insert(0, current_return)

    action_log_probs = torch.stack(action_log_probs)
    values = torch.cat(values)
    returns = torch.tensor(returns)

    # compute actor-critic loss values
    actor_loss = torch.mean(-action_log_probs * (returns - values.detach()))
    critic_loss =  F.mse_loss(values, returns, reduction = 'mean')

    return actor_loss, critic_loss, np.sum(rewards)


def train_agent(env, agent,
                gamma=0.99,
                n_epochs_warmup=500,
                n_epochs=10000,
                batch_size=1,
                max_steps_per_episode=1000,
                log_every=1000,
                lr=1e-2,
                logfile=None):

    optimizer = optim.Adam(agent.parameters(), lr=lr)

    best_running_return = -float("inf")
    best_params = agent.state_dict().copy()

    for ep in range(n_epochs+n_epochs_warmup) :

        epoch_return = 0
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        optimizer.zero_grad()

        loss = 0.

        for i in range(batch_size):
            actor_loss, critic_loss, episode_return = loss_episode(env, agent, gamma, max_steps_per_episode)

            epoch_return += episode_return / batch_size
            epoch_actor_loss += actor_loss.detach().item() / batch_size
            epoch_critic_loss += critic_loss.detach().item() / batch_size

            loss += 10 * critic_loss / batch_size

            if ep >= n_epochs_warmup:
                loss += actor_loss / batch_size

        if ep == 0:
            running_return = epoch_return
            running_critic_loss = epoch_critic_loss

        running_return = epoch_return * 0.1 + running_return * 0.9
        running_critic_loss = epoch_critic_loss * 0.1 + running_critic_loss * 0.9

        if ep % log_every == 0:
            print_log(f'Epoch {ep}: running return= {np.round(running_return, 4)}, critic loss={np.round(running_critic_loss, 4)}', logfile=logfile)

        if ep == n_epochs_warmup:
            print_log("  critic warmup complete", logfile=logfile)

        # store best agent
        if ep >= n_epochs_warmup and best_running_return < running_return:
            print_log(f"  best agent so far ({np.round(running_return, 4)})", logfile=logfile)
            best_running_return = running_return
            best_params = agent.state_dict().copy()

        loss.backward()
        optimizer.step()

    # restore best agent
    agent.load_state_dict(best_params)


def evaluate_agent(env, agent, n_episodes, max_steps_per_episode=1000):
    with torch.no_grad():
        mean_return = 0
        for ep in range(n_episodes):
            _, _, rewards = run_episode(env, agent, max_steps_per_episode)
            mean_return += np.sum(rewards) / n_episodes

    return mean_return
