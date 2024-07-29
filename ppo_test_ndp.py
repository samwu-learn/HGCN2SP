# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse, json
from itertools import product
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
from utils import solve_ndp_softmax
import gurobipy as gp
from models import Encoder, DecoderCell, CriticCell
import os
import pandas as pd
import torch.nn.functional as F

 

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """实验的名称"""
    device_id: int = 2
    """device的id 位置"""
    seed: int = 16
    """实验的随机种子"""
    torch_deterministic: bool = True
    """如果设置为True,则`torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """如果设置为True,则默认启用cuda"""
    model_test_path: str = "./model_path/NDP.pt"
    """模型测试的参数地址"""

    # 算法特定的参数
    mode: str = "test"
    """"算法模式"""
    env_id: str = "NDP"
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

def solve_ndp_scenarios(cluster, index = None, weights = None, first_precision = None):

    param = cluster['param']
    scenarios = cluster['scenarios']
    
    model = gp.Model()
    var_dict = {}

    n_origins = param['n_origins']
    n_destinations = param['n_destinations']
    n_intermediates = param['n_intermediates']
        
    # vertices
    vertices = range(n_origins + n_destinations + n_intermediates) # all vertices
    origins = vertices[:n_origins] # origins
    destinations = vertices[-n_destinations:] # destinations
        
    # arcs
    od_arcs = list(product(origins, destinations)) # all arcs linking origins to destinations
    transport_arcs = [arc for arc in product(vertices, vertices) 
                            if arc[0] != arc[1] and arc not in od_arcs] # all arcs but od pairs
    all_arcs = od_arcs + transport_arcs

    cargoes = {}
    num = 0
    for edge in od_arcs:
        cargoes[edge] = num
        num += 1

    if index is not None:
        idx = index
    else:
        idx = list(range(len(scenarios)))

    if weights is None:
        prob = 1.0 / len(scenarios)
        weights = [prob for i in range(len(scenarios))]

    # 目标函数
    # 每个边的二进制变量 第一阶段 
    for arc in transport_arcs:
        i = arc[0]
        j = arc[1]
        var_name = f"y_{i}_{j}"
        if first_precision is None:
            var_dict[var_name] = model.addVar(
                lb=0.0,
                ub=1.0,
                obj=param['opening_cost'][arc],
                vtype="B",
                name=var_name,
            )
        else :
            var_dict[var_name] = model.addVar(   #固定第一阶段决策变量
                lb=first_precision[f"y_{i}_{j}"],
                ub=first_precision[f"y_{i}_{j}"],
                obj=param['opening_cost'][arc],
                vtype="B",
                name=var_name,
            )


    # 第二阶段
    loc = 0
    for s in idx:
        for (arc, od) in product(all_arcs, od_arcs):
            i = arc[0]
            j = arc[1]
            k = cargoes[od]
            var_name = f"z_{i}_{j}_{k}_{s}"
            var_dict[var_name] = model.addVar(
                lb=0.0,
                obj=param['shipping_cost'][arc] * weights[loc],
                vtype="C",
                name=var_name,
            )
        loc = loc + 1

    # 约束条件
    #约束1  transport_constraints
    for s in idx:
        for arc in transport_arcs:
            i = arc[0]
            j = arc[1]
            cons = - param['capacity'][arc] * var_dict[f"y_{i}_{j}"]
            for od in od_arcs:
                k = cargoes[od]
                cons += var_dict[f"z_{i}_{j}_{k}_{s}"]
            model.addConstr(cons<=0, name = f"open_{i}_{j}_{s}")

    def d(vertex, od, s):
        k = cargoes[od]
        demand = scenarios[s][k]
        if vertex == od[0]: # if the vertex is the origin of the commodity
            return demand
        elif vertex == od[1]: # if the vertex is the destination of the commodity
            return -demand
        else: # if the vertex is an intermediate step
            return 0

    #约束2  demand_constraints
    for s in idx:
        for vertex, od in product(vertices, od_arcs):
            k = cargoes[od]
            cons = 0
            for arc in all_arcs:
                i = arc[0]
                j = arc[1]
                if arc[0] == vertex:
                    cons += var_dict[f"z_{i}_{j}_{k}_{s}"]
                elif arc[1] == vertex:
                    cons += -var_dict[f"z_{i}_{j}_{k}_{s}"]
                else :
                    continue
            demand = d(vertex, od, s)
            model.addConstr(cons==demand, name = f"demand_{vertex}_{od[0]}_{od[1]}_{s}")
    
    model.update()

    #set param
    model.setParam('OutputFlag', 0) 
    model.setParam("MIPGap", 0.00)
    model.setParam("TimeLimit", 10800)
    # 设置线程数
    model.setParam("Threads", 16)

    model.optimize()

    # 获取第一阶段变量的取值 
    variable_values = {var.varName: var.x for var in model.getVars() if 'y' in var.varName}

    solving_results = {}
    solving_results["primal"] = model.objVal
    solving_results["time"] = model.Runtime
    solving_results['X'] = variable_values

    return solving_results


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

def test_model(test_cls_path, action, test_ins_num, test_pararm, test_type, weights):
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
        with open(file_path, "rb") as f:
            results = pickle.load(f)
            bs = results['primal']
            mean_bs += bs
            np_bs.append(bs)
        index = action[i].cpu().numpy()
        weight = weights[i]
        test_results = solve_ndp_scenarios(cls, index, weight)
        test_time = test_results['time']
        if test_type:
            print("fixed first decision_stage")
            test_results = solve_ndp_scenarios(cls, first_precision=test_results['X'])
        test_results['time'] = test_time
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
    print('type:',type(mean_agent),type(mean_bs),type(mean_time))
    return file_name,np_agent, np_bs, np_time, np_delta

def test(parser, model_test_path, test_ins_num, device,test_type='Non_first',nums=2, seed=1234):
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

    print("save_path:", test_pararm['save_path'])

    test_data = torch.load(test_pararm['save_path'])
    

    print(len(test_data))
    test_decoder = 'greedy'
    sel_num = all_kwargs['test']['sel_num']
    print("sel_num:", sel_num)
    test_data = [test_data[i] for i in range(test_ins_num)]
    file_name_all = []
    with open(test_pararm['cls_path'], 'rb') as f:
        test_cls_path = pickle.load(f)
    idxs, ag_data, bs_data, ti_data, da_data = [],[],[],[],[]
    print("test_type:",test_type)
    for num in range(nums): 
        with torch.no_grad():
            start = time.time()
            action, log_p= agent.get_action_and_value(test_data, decode_type=test_decoder, cluster_k=sel_num)
            print("Model time:", time.time()-start)
            log_p = F.softmax(log_p.squeeze(), dim = -1)
            weights = log_p.cpu().numpy()
    
            file_name, np_agent, np_bs, np_time, np_delta = test_model(test_cls_path, action, test_ins_num, test_pararm, test_type, weights)
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
    'gap': da_data
    })
    csv_file_path = f'./test_csv/ppo_test_results_best_ndp_k_{sel_num}_seed_{seed}.csv'  # 修改为您希望保存文件的路径
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
    parser.add_argument('--config_file', type=str, default='./configs/ndp_config_norm.json', help="base config json dir")

    args = tyro.cli(Args)
    print("seed:",args.seed)
        # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.device_id is not None:
        torch.cuda.set_device(args.device_id)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    test(parser, args.model_test_path, 200, device, nums=1, seed = args.seed)
