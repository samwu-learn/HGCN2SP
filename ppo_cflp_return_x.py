# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse, json
import math
import os
import pickle
import random
import re
import time
from dataclasses import dataclass
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
import torch.multiprocessing as mp
from utils import solve_cflp_softmax_new, solve_cflp_softmax
from models import Encoder, DecoderCell, CriticCell
import os
 


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """实验的名称"""
    device_id: int = 3
    """device的id 位置"""
    seed: int = 16
    """实验的随机种子"""
    torch_deterministic: bool = True
    """如果设置为True,则`torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """如果设置为True,则默认启用cuda"""
    track: bool = True
    """如果设置为True,则使用Weights and Biases跟踪此实验"""
    wandb_project_name: str = ""
    """Weights and Biases的项目名称"""
    wandb_entity: str = ""
    """Weights and Biases的实体(团队)"""
    capture_video: bool = False
    """是否捕获代理的表现视频（请查看`videos`文件夹）"""
    save_model: bool = True
    """是否保存模型到`runs/{run_name}`文件夹"""
    upload_model: bool = False
    """是否将保存的模型上传到Hugging Face"""
    hf_entity: str = ""
    """来自Hugging Face Hub的模型仓库的用户或组织名称"""
    model_test_path: str = ""
    """模型测试的参数地址"""

    # 算法特定的参数
    mode: str = "train"
    """"算法模式"""
    env_id: str = "CFLP"
    """环境的ID"""
    total_timesteps: int = 20480
    """实验的总步数"""
    learning_rate: float = 2.5e-4
    """优化器的学习率"""
    num_envs: int = 2048
    """并行游戏环境的数量"""
    num_steps: int = 1
    """每个策略回合在每个环境中运行的步数"""
    anneal_lr: bool = True
    """是否对策略和值网络进行学习率退火"""
    gamma: float = 0.99
    """折扣因子 gamma"""
    gae_lambda: float = 0.95
    """用于广义优势估计的 lambda 值"""
    num_minibatches: int = 16
    """mini-batch 的数量"""
    update_epochs: int = 10
    """更新策略的 K 个epochs"""
    norm_adv: bool = True
    """是否进行优势归一化"""
    clip_coef: float = 0.20
    """策略梯度裁剪系数"""
    clip_vloss: bool = True
    """是否对值函数使用裁剪损失，根据论文"""
    ent_coef: float = 0.0
    """熵的系数"""
    vf_coef: float = 0.5
    """值函数的系数"""
    max_grad_norm: float = 0.5
    """梯度裁剪的最大范数"""
    target_kl: float = None
    """目标KL散度阈值"""

    # 在运行时填充
    batch_size: int = 0
    """批量大小（在运行时计算）"""
    minibatch_size: int = 0
    """mini-batch 大小（在运行时计算）"""
    num_iterations: int = 0
    """迭代次数（在运行时计算）"""

def load_param(parser, device):
    args = parser.parse_args()
    all_kwargs = json.load(open(args.config_file, 'r'))
    data_param = all_kwargs['TrainData']
    #load train param
    policy_param = all_kwargs['Policy']
    train_param = all_kwargs['train']
    data = torch.load(data_param["save_path"])
    train_cls = data_param['cls_path']
    n_scenarios = train_param["sel_num"]
    bs = []
    clusters = []
    # 从文件中读取列表
    with open(train_cls, 'rb') as f:
        cls_path = pickle.load(f)

    for i in range(len(cls_path)):
        cls_loc = os.path.join(data_param['pkl_folder'], cls_path[i])  
        with open(cls_loc, 'rb') as f:
            cls = pickle.load(f)
            clusters.append(cls)

        file_path = f"result_of_{cls_path[i][10:-4]}.pkl"
        file_path = os.path.join(data_param['result_folder'], file_path) 
        with open(file_path, "rb") as f:
            result = pickle.load(f)
            ans = []
            for key in result['X'].keys():
                ans.append(float(result['X'][key]))
            ans.append(result['primal'])
            bs.append(ans)

    return policy_param, train_param, data, bs, n_scenarios, clusters


class CFLPEnv():
    """
    CFLPEnv Environment 
    """
    def __init__(self, data, bs, clusters, k, batch_size, process = 2, device = None):
        super(CFLPEnv, self).__init__()

        # features  -> list[edge batch_data] 
        self.data = data
        self.used = list(range(len(data))) 

        # EF costs  -> list
        self.bs = bs
        # problem dict -> list
        self.clusters = clusters
        # k scenarios  -> action space size
        self.k = k
        # batch_size  -> num_envs
        self.batch_size = batch_size
        #process ->solve gurobi
        self.process = process

        self.device = device


    def choose_random_element(self):
        return self.used[self.loc]

    def reset(self, seed = None):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.used)
        self.loc = 0
        self.batch_data = [] 
        self.batch_bs = []
        self.batch_clusters = []
        for start in range(0, len(self.data), self.batch_size):
            end = start + self.batch_size
            if end >= len(self.data):
                end = len(self.data)
            self.batch_data.append([self.data[i] for i in self.used[start:end] ])
            self.batch_bs.append([self.bs[i] for i in self.used[start:end] ])
            self.batch_clusters.append([self.clusters[i] for i in self.used[start:end] ])
        return self.batch_data[self.loc], {}
    
    def rein_reward(self, action, iteration, alpha = 0.001):
        # 将数据分割成batch个子列表，每个子列表交给一个进程处理
        action = torch.from_numpy(action)
        chunks = action.unbind(0)
        param_values = self.batch_clusters[self.loc]
        is_trains = [True] * self.batch_size
        # 创建进程池
        with mp.Pool(processes = self.process) as pool:
            # 使用进程池进行并行处理，传递包含多个参数的元组
            results = pool.map(solve_cflp_softmax_new, zip(param_values, chunks, is_trains))	
        #合并结果
        rewards = torch.stack(results)
        base_cost = torch.from_numpy(np.array(self.batch_bs[self.loc]))
        # 计算cost奖励
        cost = torch.sum((base_cost[:, :-1] == rewards[:,:-2]), dim=1)/(rewards.shape[-1]-2)
        # 计算time奖励
        time = rewards[:, -2]
        
        # 求和得到结果
        rewards = 25 * (cost - alpha *time)
        return rewards.numpy()

    def step(self, action, iteration):
        reward = self.rein_reward(action, iteration)
        terminations, truncations = np.array([True]*self.batch_size), np.array([True]*self.batch_size)
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        self.loc = self.loc + 1
        if self.loc >= len(self.batch_data):
            obs, _ = self.reset()
        else:
            obs = self.batch_data[self.loc]
        return obs, reward,  terminations, truncations, info

    def close(self):
        print("EndEnv...")
        pass



class Agent(nn.Module):
    def __init__(self, policy_param, train_param, device):
        super().__init__()
        self.features_extractor = Encoder(policy_param['var_dim'], policy_param['con_dim'], policy_param['l_hid_dim'], policy_param['scenario_dim'], policy_param['h_hid_dim'], policy_param['h_out_dim'])
        self.actor = DecoderCell(policy_param['scenario_dim'], policy_param['n_heads'], policy_param['clip'])
        self.critic = CriticCell()
        self.device = device
        self.decode_type = train_param['decode_type']
        self.cluster_k = train_param['sel_num']


    def get_action_and_value(self, x, action=None, decode_type="sampling", cluster_k=None):
        self.decode_type = decode_type
        batch_feat = []
        batch_edge = []
        for i in range(len(x)): #(batch 200xn dim )
            x[i] = x[i].to(device)
            feat, c = self.features_extractor(x[i])
            batch_feat.append(feat)
            batch_edge.append(c)
            x[i] = x[i].cpu()
        batch_feat = torch.stack(batch_feat)
        batch_edge = torch.stack(batch_edge)
        encoder_output = (batch_feat, batch_edge)
        if action is not None:
            action = action.to(torch.int64)
        if cluster_k is not None:
            self.cluster_k = cluster_k
        action, logprob, entropy = self.actor(self.device, encoder_output, return_cost = False, cluster_k = self.cluster_k, decode_type = self.decode_type, action = action)
        value = self.critic(batch_edge, batch_feat)
        
        return action, logprob, entropy, value

def eval_model(eval_cls_path, action, agent, eval_epoch, best_eval_delta, run_name):
    mean_bs = 0
    mean_agent = 0
    mean_time = 0
    delta = 0
    eval_num = min(100, len(eval_cls_path))
    for i in range(eval_num):
        cls_loc = os.path.join("./scenarios_origin", eval_cls_path[i]) 
        with open(cls_loc, 'rb') as f:
            cls = pickle.load(f)
        file_path = f"result_of_{eval_cls_path[i][10:-4]}.pkl"
        file_path = os.path.join("./new_resu", file_path)
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
    if (delta) < best_eval_delta:
        print("Saving models...")
        best_eval_delta = (delta)
        model_path = f"./model_path/{run_name}_eval_{eval_epoch}_{round(best_eval_delta, 2)}.pt"  #p_change
        torch.save(agent.state_dict(), model_path)
    eval_epoch += 1
    print(f"Eval  Averge:  agent:{mean_agent}  bs:{mean_bs}  time:{mean_time}  delta:{delta}%")
    wandb.log({"mean_agent": mean_agent,  "Averge bs": mean_bs, "time": mean_time, "delta": delta})
    return best_eval_delta

def test_model(test_cls_path, action):
    mean_bs = 0
    mean_agent = 0
    mean_time = 0
    delta = 0
    for i in range(len(test_cls_path)):
        cls_loc = os.path.join("./scenarios_new", test_cls_path[i]) #p_change
        with open(cls_loc, 'rb') as f:
            cls = pickle.load(f)
        file_path = f"result_of_{test_cls_path[i][10:-4]}.pkl"
        file_path = os.path.join("./results_new", file_path)   #p_change
        with open(file_path, "rb") as f:
            results = pickle.load(f)
            bs = results['primal']
            mean_bs += bs
        args = (cls, action[i].cpu(), True)
        test_results = solve_cflp_softmax(args).squeeze()
        now_delta = (test_results[0].item() - bs)/ bs *100
        print(f"Test {i}: {test_results[0].item()} , {test_results[1].item()} , bs: {bs} , delta: {now_delta} %")
        mean_agent += test_results[0].item()
        mean_time += test_results[1].item()
        delta += abs((test_results[0].item() - bs)/ bs *100)

    mean_agent = mean_agent / len(test_cls_path)
    mean_bs = mean_bs / len(test_cls_path)
    mean_time = mean_time / len(test_cls_path)
    delta /= len(test_cls_path)
    print(f"Test  Averge:  agent:{mean_agent}  bs:{mean_bs}  time:{mean_time}  delta:{delta}%")


if __name__ == "__main__":
    current_dir = os.getcwd()
    print("current_dir:",current_dir)
    # 参数配置：固定参数json 文件；调试参数命令行
    parser = argparse.ArgumentParser(description="Gnn_Transformer for two_stage")
    parser.add_argument('--config_file', type=str, default='./configs/cflp_config.json', help="base config json dir")

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track and args.mode == "train":
        

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.device_id is not None:
        torch.cuda.set_device(args.device_id)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    

    policy_param, train_param, data, bs, n_scenarios, clusters = load_param(parser, device)

    # env setup
    envs = CFLPEnv(data, bs, clusters, n_scenarios, args.num_envs, device = device)

    agent = Agent(policy_param, train_param, device).to(device)

    if args.mode == "test":
        print("Test....")
        param = torch.load(args.model_test_path, map_location=device)
        print("load...")
        agent.load_state_dict(param)
        console_args = parser.parse_args()
        all_kwargs = json.load(open(console_args.config_file, 'r'))
        test_pararm = all_kwargs['TestData']
        test_data = torch.load(test_pararm['save_path'])
        test_decoder = all_kwargs['test']['decode_type']
        sel_num = all_kwargs['test']['sel_num']
        for i in range(len(test_data)):
            test_data[i] = test_data[i].to(device)
        with open(test_pararm['cls_path'], 'rb') as f:
            test_cls_path = pickle.load(f)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(test_data, decode_type=test_decoder, cluster_k=sel_num)
            print(action)
            test_model(test_cls_path, action)
        os._exit(0)



    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    #obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device) #change
    obs = []
    #actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, envs.k)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)  #change
    #next_obs = torch.Tensor(next_obs).to(device)  #change
    next_done = torch.zeros(args.num_envs).to(device)

    eval_epoch, best_eval_delta = 0, 100.

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        obs = []
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs.append(next_obs)
            #obs[step] = next_obs  #change
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, siid, value = agent.get_action_and_value(next_obs)  #change
                # print(action.shape, logprob.shape, siid.shape, value.shape)
                # os._exit(0)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy(),iteration)  #change
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            #next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device) #change
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        # bootstrap value if not done
        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                delta = rewards[t] - values[t] 
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        #b_obs = obs.reshape((-1,) + envs.single_observation_space.shape) #change
        b_obs = [element for sublist in obs for element in sublist]
        b_logprobs = logprobs.reshape(-1)
        #b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_actions = actions.reshape((-1, envs.k))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                b_obs_m = [b_obs[i] for i in mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs_m, b_actions[mb_inds]) #change
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                print("loss:", loss.item())
                wandb.log({'epoch': epoch, 'loss': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        if iteration % train_param['eval_epoch'] == 0:
            torch.save(agent.state_dict(), f"./model_path/{run_name}_seed_{args.seed}_{iteration}.pt")
            print("Eval...")
            eval_data = torch.load(os.path.join(train_param['eval_path'],"eval_origin.pt"))   
            for i in range(len(eval_data)):
                eval_data[i] = eval_data[i].to(device)
            with open(os.path.join(train_param['eval_path'],"eval_cls_origin.pkl"), 'rb') as f:   
                eval_cls_path = pickle.load(f)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(eval_data, decode_type="greedy")
                best_eval_delta = eval_model(eval_cls_path, action, agent, eval_epoch, best_eval_delta, run_name)
                eval_epoch = eval_epoch + 1

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print("SPS:", int(global_step / (time.time() - start_time)))

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.pt"
        # 提取文件夹路径
        folder_path = os.path.dirname(model_path)
        # 检查文件夹是否存在，如果不存在则创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    envs.close()
