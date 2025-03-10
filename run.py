# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from dataclasses import dataclass
import os
import time
import torch
import numpy as np
import wandb
from agent import Agent
from env import CFLPEnv
from sample import Sampler
import argparse
import json
import pickle
import random
import pandas as pd
import tyro
from utils import solve_cflp_softmax_new
from trainer import PPOTrainer

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
    track: bool = False
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

# 读取参数
def load_param(parser, device, mode = "train"):
    args = parser.parse_args()
    all_kwargs = json.load(open(args.config_file, 'r'))
    data_param = all_kwargs['TrainData']
    #load train param
    policy_param = all_kwargs['Policy']
    train_param = all_kwargs['train']
    if mode == "train":
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
    elif mode == "test":
        return policy_param, train_param
    else:
        raise ValueError("mode should be train or test")

# 测试用函数
def test_model(test_cls_path, action, test_ins_num, test_pararm):
    mean_bs = 0
    mean_agent = 0
    mean_time = 0
    delta = 0
    np_bs, np_agent, np_time, np_delta = [],[],[],[]
    file_name = []
    for i in range(test_ins_num):
        cls_loc = os.path.join(test_pararm['pkl_folder'], test_cls_path[i])
        file_name.append(cls_loc)
        with open(cls_loc, 'rb') as f:
            cls = pickle.load(f)
        
        file_path = f"result_of_{test_cls_path[i][10:-4]}.pkl"
        file_path = os.path.join(test_pararm['result_folder'], file_path)  
        bs_x = []
        with open(file_path, "rb") as f:
            results = pickle.load(f)
            bs = results['primal']
            for key in results['X'].keys():
                bs_x.append(results['X'][key])
            mean_bs += bs
            np_bs.append(bs)
        bs_x = torch.from_numpy(np.array(bs_x))
        args = (cls, action[i].cpu(), True)
        test_ = solve_cflp_softmax_new(args).squeeze()   
        cost = torch.sum((bs_x == test_[:-2]), dim=0)/(test_.shape[-1]-2)
        test_results = {}
        test_results['primal'] = test_[-1].item()
        test_results['time'] = test_[-2].item()
        if abs(cost - 1.0) < 1e-5: 
            now_delta = 0.0
        else :
            now_delta = (test_results['primal'] - bs)/ bs *100
        print(f"Test {i}  {test_cls_path[i]}: {test_results['primal']} , {test_results['time']} , bs: {bs} , gap: {now_delta} %")

        mean_agent += test_results['primal']
        np_agent.append(test_results['primal'])
        mean_time += test_results['time']
        np_time.append(test_results['time'])
        delta += ((test_results['primal'] - bs)/ bs *100)
        np_delta.append(((test_results['primal'] - bs)/ bs *100))
    mean_agent = mean_agent / test_ins_num
    mean_bs = mean_bs / test_ins_num
    mean_time = mean_time / test_ins_num
    delta /= test_ins_num
    print(f"Test  Averge:  agent:{mean_agent}  bs:{mean_bs}  time:{mean_time}  gap:{delta}%")
    return file_name,np_agent, np_bs, np_time, np_delta

def test(parser, mode, model_test_path, test_ins_num, device, nums=1, seed=16, clip=0.2, alpha=0.001):
    '''num: 测试的次数，多次试验，避免机器、随机性的影响'''
    mean_agent, mean_bs, mean_time, delta = [],[],[],[]
    policy_param, train_param = load_param(parser, device, mode = mode)
    agent = Agent(policy_param, train_param, device).to(device)
    print("Test "+model_test_path+" ....")
    param = torch.load(model_test_path, map_location=device)
    agent.load_state_dict(param)
    console_args = parser.parse_args()
    all_kwargs = json.load(open(console_args.config_file, 'r'))
    test_pararm = all_kwargs['TestData']
    test_data = torch.load(test_pararm['save_path'])
    test_ins_num = min(test_ins_num, len(test_data))

    test_decoder = all_kwargs['test']['decode_type']
    sel_num = all_kwargs['test']['sel_num']
    print("Select numbers:",sel_num)
    test_data = [test_data[i] for i in range(test_ins_num)]
    file_name_all = []
    with open(test_pararm['cls_path'], 'rb') as f:
        test_cls_path = pickle.load(f)
    idxs, ag_data, bs_data, ti_data, da_data = [],[],[],[],[]
    for num in range(nums): 
        with torch.no_grad():
            start = time.time()
            action, log_p= agent.get_action_and_value(test_data, decode_type=test_decoder, cluster_k=sel_num, mode = mode)
            print("Model_time:", time.time() - start)
            file_name, np_agent, np_bs, np_time, np_delta = test_model(test_cls_path, action, test_ins_num, test_pararm)
            idx = [num]*len(np_agent)
            ag,bs,ti,de = np.mean(np_agent),np.mean(np_bs),np.mean(np_time),np.mean(np_delta)
        file_name_all.extend(file_name)
        ag_data.extend(np_agent)
        bs_data.extend(np_bs)
        ti_data.extend(np_time),
        da_data.extend(np_delta)
        idxs.extend(idx)
        mean_agent.append(ag)
        mean_bs.append(bs)
        mean_time.append(ti)
        delta.append(de)
    results_df = pd.DataFrame({
    'idx': idxs,
    'file_name': file_name_all,
    'real_results': bs_data,
    'model_result': ag_data,
    'model_time': ti_data,
    'gap': da_data,
    'error':da_data
    })
    # print(results_df)
    csv_file_path = f'./test_csv/ppo_test_results_cflp_10_20_{sel_num}_clip_{clip}_alpha_{alpha}.csv'  # 修改为您希望保存文件的路径
    results_df.to_csv(csv_file_path, index=False)
    print("results_df:", results_df)
    # 输出保存成功的消息
    print(f"Results have been saved to {csv_file_path}")
    
    print('test done for '+model_test_path +' with '+str(nums)+' times: ')
    print('mean_agent: ',np.mean(mean_agent),' mean_bs: ',np.mean(mean_bs), 
          ' mean_time: ',np.mean(mean_time), ' delta: ',np.mean(delta))


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

    if args.mode == "train":
        policy_param, train_param, data, bs, n_scenarios, clusters = load_param(parser, device)
        # trainer setup
        trainer = PPOTrainer(args, policy_param, train_param, data, bs, clusters, run_name, device)
        trainer.train()
    elif args.mode == "test":
        test(parser, args.mode, args.model_test_path, 100, device, nums=1, seed = args.seed, clip = args.clip_coef, alpha = 0.0001)

        