import torch
import numpy as np

class Sampler:
    def __init__(self, envs, agent, num_steps, device):
        self.envs = envs
        self.agent = agent
        self.num_steps = num_steps
        self.device = device
        self.reset_buffer()

    def reset_buffer(self):
        self.obs = []
        self.actions = torch.zeros((self.num_steps, self.envs.batch_size, self.envs.k), device=self.device)
        self.logprobs = torch.zeros((self.num_steps, self.envs.batch_size), device=self.device)
        self.rewards = torch.zeros((self.num_steps, self.envs.batch_size), device=self.device)
        self.values = torch.zeros((self.num_steps, self.envs.batch_size), device=self.device)
        self.next_done = torch.zeros(self.envs.batch_size, device=self.device)

    def collect_trajectories(self, next_obs):
        for step in range(self.num_steps):
            self.obs.append(next_obs)
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
            next_obs, reward, terminations, truncations, _ = self.envs.step(action.cpu().numpy(), step)
            next_done = torch.Tensor(np.logical_or(terminations, truncations)).to(self.device)
            
            # 存储数据
            self.actions[step] = action
            self.logprobs[step] = logprob
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            self.values[step] = value.flatten()
            self.next_done = next_done
        return next_obs, self.next_done

    def compute_advantages_and_returns(self, args):
        with torch.no_grad():
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                delta = self.rewards[t] - self.values[t] 
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values
        return advantages, returns

    def get_batch_data(self, advantages, returns):
        b_obs = [element for sublist in self.obs for element in sublist]
        b_actions = self.actions.reshape((-1, self.envs.k))
        b_logprobs = self.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        return b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values