import os
import time
import torch
import numpy as np
import wandb
import pickle
import torch.nn as nn  
from agent import Agent
from env import CFLPEnv
from sample import Sampler
from utils import solve_cflp_softmax

class PPOTrainer:
    def __init__(self, args, policy_param, train_param, data, bs, clusters, run_name, device):
        self.args = args
        self.train_param = train_param
        self.run_name = run_name
        self.device = device
        self.envs = CFLPEnv(data, bs, clusters, train_param['sel_num'], args.num_envs, device=self.device)
        self.agent = Agent(policy_param, train_param, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)
        self.sampler = Sampler(self.envs, self.agent, args.num_steps, self.device)
        self.best_eval_delta = 100.0
        self.eval_epoch = 0
        self.use_wandb = args.track and hasattr(wandb, 'run') and wandb.run is not None

    def train(self):
        next_obs, _ = self.envs.reset(seed=self.args.seed)
        next_done = torch.zeros(self.envs.batch_size, device=self.device)

        for iteration in range(1, self.args.num_iterations + 1):
            if self.args.anneal_lr:
                self._anneal_learning_rate(iteration)

            # 数据采集
            next_obs, next_done = self.sampler.collect_trajectories(next_obs)

            # 计算优势与回报
            advantages, returns = self.sampler.compute_advantages_and_returns(self.args)

            # 获取批数据
            b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values = self.sampler.get_batch_data(advantages, returns)

            # 策略优化
            self._update_policy(b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values)

            # 评估与保存模型
            if iteration % self.train_param['eval_epoch'] == 0:
                self._evaluate_and_save(iteration)
        
        # 训练完成后保存模型 
        if self.args.save_model:
            model_path = f"runs/{self.run_name}/{self.args.exp_name}.pt"
            # 提取文件夹路径
            folder_path = os.path.dirname(model_path)
            # 检查文件夹是否存在，如果不存在则创建
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            torch.save(self.agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        self.envs.close()

    def _update_policy(self, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values):
        # Optimizing the policy and value network
        b_inds = np.arange(self.args.batch_size)
        clipfracs = []
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                b_obs_m = [b_obs[i] for i in mb_inds]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs_m, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                print("loss:", loss.item())

                if self.use_wandb:
                    wandb.log({'epoch': epoch, 'loss': loss.item()})
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                break

    def _anneal_learning_rate(self, iteration):
        frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
        self.optimizer.param_groups[0]["lr"] = frac * self.args.learning_rate
    
    def eval_model(self, eval_cls_path, action, train_param):
        mean_bs = 0
        mean_agent = 0
        mean_time = 0
        delta = 0
        eval_num = min(100, len(eval_cls_path))
        for i in range(eval_num):
            cls_loc = os.path.join(train_param["eval_cls_loc"], eval_cls_path[i]) 
            with open(cls_loc, 'rb') as f:
                cls = pickle.load(f)
            file_path = f"result_of_{eval_cls_path[i][10:-4]}.pkl"
            file_path = os.path.join(train_param["eval_result"], file_path)
            with open(file_path, "rb") as f:
                results = pickle.load(f)
                bs = results['primal']
                mean_bs += bs
            args = (cls, action[i].cpu(), True)
            eval_results = solve_cflp_softmax(args).squeeze()
            print(eval_results, bs, (eval_results[0].item() - bs)/ bs *100)
            mean_agent += eval_results[0].item()
            mean_time += eval_results[1].item()
            delta += ((eval_results[0].item() - bs)/ bs *100)

        mean_agent = mean_agent / eval_num
        mean_bs = mean_bs / eval_num
        mean_time = mean_time / eval_num
        delta /= eval_num
        if (delta) < self.best_eval_delta:
            print("Saving models...")
            self.best_eval_delta = (delta)
            model_path = f"./model_path/{self.run_name}_eval_{self.eval_epoch}_{round(self.best_eval_delta, 2)}.pt"
            torch.save(self.agent.state_dict(), model_path)
        self.eval_epoch += 1
        print(f"Eval  Averge:  agent:{mean_agent}  bs:{mean_bs}  time:{mean_time}  delta:{delta}%")
        
        if self.use_wandb:
            wandb.log({"mean_agent": mean_agent, "Averge bs": mean_bs, "time": mean_time, "delta": delta})
    
    def _evaluate_and_save(self, iteration):
        train_param = self.train_param
        # Evaluate 自行修改存储位置
        torch.save(self.agent.state_dict(), f"./model_path/{self.run_name}_seed_{self.args.seed}_{iteration}.pt")
        print("Eval...")
        eval_data = torch.load(os.path.join(train_param['eval_path'], train_param['eval_pt']))   
        for i in range(len(eval_data)):
            eval_data[i] = eval_data[i].to(self.device)
        with open(os.path.join(train_param['eval_path'], train_param['eval_cls']), 'rb') as f:   
            eval_cls_path = pickle.load(f)
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(eval_data, decode_type="greedy")
            self.eval_model(eval_cls_path, action, train_param)