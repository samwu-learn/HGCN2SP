# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse, json
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
from utils import solve_cflp_softmax_new
from models import Encoder, DecoderCell, CriticCell, Encoder
import os
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """实验的名称"""
    device_id: int = 2
    """device的id 位置"""
    seed: int = 4
    """实验的随机种子"""
    torch_deterministic: bool = True
    """如果设置为True,则`torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """如果设置为True,则默认启用cuda"""
    model_test_path: str = "./model_path/CFLP.pt"
    """模型测试的参数地址"""

    # 算法特定的参数
    mode: str = "test"
    """"算法模式"""
    env_id: str = "CFLP"
    """环境的ID"""
    clip_coef: float = 0.25
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



def load_param(parser):
    args = parser.parse_args()
    all_kwargs = json.load(open(args.config_file, 'r'))
    #load train param
    policy_param = all_kwargs['Policy']
    train_param = all_kwargs['train']

    return policy_param, train_param



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
        #print("node_em_of_different_scenarioss:", torch.mean(x[0].x[0:251] - x[0].x[251:502]))
        for i in range(len(x)):
            x[i] = x[i].to(self.device)
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
        action, logprob = self.actor(self.device, encoder_output, return_cost = True, cluster_k = self.cluster_k, decode_type = self.decode_type, action = action)
        
        return action, logprob

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

def test(parser, model_test_path, test_ins_num, device, nums=1, seed=16, clip=0.2, alpha=0.001):
    '''num: 测试的次数，多次试验，避免机器、随机性的影响'''
    mean_agent, mean_bs, mean_time, delta = [],[],[],[]
    policy_param, train_param = load_param(parser)
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
            action, log_p= agent.get_action_and_value(test_data, decode_type=test_decoder, cluster_k=sel_num)
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
    # import sys 
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.device_id is not None:
        torch.cuda.set_device(args.device_id)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    test(parser, args.model_test_path, 100, device, nums=1, seed = args.seed, clip = args.clip_coef, alpha = 0.0001)