from itertools import product
import pickle
import time
import torch
import argparse, os, json
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os,pickle
from scipy.sparse import load_npz, lil_matrix
from utils.processer import normalize_adj, sparse_mx_to_torch_sparse_tensor, preprocess_features
from utils.cflpdata import CustomDataset
from scipy.sparse import csr_matrix
import pickle
from torch_geometric.data import Data, Batch
from multiprocessing import Manager, Pool


#文件排序  --------------------------
def sort_key(file_name):
    start_index = file_name.rindex("_") + 1  # 获取标号的起始索引
    end_index = file_name.index(".")  # 获取标号的结束索引
    label = int(file_name[start_index:end_index])  # 提取标号并转换为整数
    return label


#提取问题特征  --------------------------
def _get_cflp_features(cluster_dict):
        #获取基本数据 顾客数 设施数 以及 场景数
    n_customers = cluster_dict["n_customers"]
    n_facilities = cluster_dict["n_facilities"]
    n_scenarios = cluster_dict["n_scenarios"]

    adjs = []
    all_feature = None
    cons_feature = None
    cflp_feature = []    #  标记某个场景的特征
    # 使用 get 方法判断键 'd' 是否存在
    if cluster_dict["second_stage"].get('z_coeff') is None:
        cluster_dict["second_stage"]['z_coeff'] = 1

    #分场景保存数据
    for s in range(n_scenarios):
        con_index = 0
        cflp_feature = []  #更新特征
        data = []
        row = []
        col = []

        cons_number = n_customers + n_facilities + 1
        #约束阶段
        #第一阶段约束 对设施数的限制
        cons = []
        cons_a = []
        cons_b = []
        for j in range(n_facilities):
            cons_a.append(1)
            cons_b.append(cluster_dict["first_stage"]["facilities_cost"][j])
            #邻接矩阵
            data.append(1)
            row.append(con_index)
            col.append(cons_number+j)
        
        cons_a.append(-cluster_dict['first_stage']['constraint'])
        cons_b.append(0)
        cons_a = np.array(cons_a)
        cons_b = np.array(cons_b)
        obj = np.dot(cons_a,cons_b) / (np.linalg.norm(cons_a)*np.linalg.norm(cons_b))
        cons.append(obj)
        cflp_feature.append(cons)

        con_index += 1


        #第二阶段约束 注意每个场景都有以下两个约束
        #约束 1 需求约束
        for j in range(n_facilities):
            cons = []
            cons_a = []
            cons_b = []
            cons_a.append((-1.0)*cluster_dict['second_stage']["qx"][j])  #xj
            cons_a.append(cluster_dict["second_stage"]['z_coeff']) #zj cluster_dict["second_stage"]["recourse_costs"][s]
            cons_b.append(cluster_dict["first_stage"]["facilities_cost"][j])
            cons_b.append(cluster_dict["second_stage"]["recourse_costs"])
            for i in range(n_customers): #yij
                cons_a.append(cluster_dict['second_stage']["q_c_f"][s][i][j])
                cons_b.append(cluster_dict["second_stage"]["trans_cost"][i][j])
                #邻接矩阵
                data.append(cluster_dict['second_stage']["q_c_f"][s][i][j])
                row.append(con_index)
                col.append(cons_number+n_facilities+i*n_facilities+j)
            cons_a = np.array(cons_a)
            cons_b = np.array(cons_b)
            obj = np.dot(cons_a,cons_b) / (np.linalg.norm(cons_a)*np.linalg.norm(cons_b))
            cons.append(obj)
            cflp_feature.append(cons)

            #邻接矩阵
            data.append((-1.0)*cluster_dict['second_stage']["qx"][j])
            row.append(con_index)
            col.append(cons_number+j)
            data.append(cluster_dict["second_stage"]['z_coeff']) #cluster_dict["second_stage"]["recourse_costs"][s]
            row.append(con_index)
            col.append(cons_number+n_facilities+n_facilities*n_customers+j)

            con_index += 1

        #约束2 存在约束
        for i in range(n_customers):
            cons = []
            cons_a = []
            cons_b = []
            cons_a.append((-1)*cluster_dict['second_stage']["h"][s][i]) #hsi
            cons_b.append(0)
            for j in range(n_facilities):
                cons_a.append(1)
                cons_b.append(cluster_dict["second_stage"]["trans_cost"][i][j])
                #邻接矩阵
                data.append(1)
                row.append(con_index)
                col.append(cons_number+n_facilities+i*n_facilities+j)
            cons_a = np.array(cons_a)
            cons_b = np.array(cons_b)
            obj = np.dot(cons_a,cons_b) / (np.linalg.norm(cons_a)*np.linalg.norm(cons_b))
            cons.append(obj)
            cflp_feature.append(cons)

            con_index += 1
        
        #生成约束的特征以及adj
        features = np.array(cflp_feature)
        features = lil_matrix(features.astype(float))
        #features, _ = preprocess_features(features)
        features = features.todense()
        features = torch.FloatTensor(features[np.newaxis])
        if cons_feature == None:
            cons_feature = features
        else :
            cons_feature = torch.cat((cons_feature, features), dim = 0)
        
        feature_size = len(cflp_feature) + n_facilities + n_facilities * n_customers  + n_facilities
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        adj = csr_matrix((data, (row, col)), shape=(feature_size, feature_size))
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adjs.append(adj)
        
        cflp_feature = []
        # xj
        for j in range(n_facilities):
            var = []
            var.append(cluster_dict["first_stage"]["facilities_cost"][j])
            var.append(cluster_dict['second_stage']["qx"][j])
            cflp_feature.append(var)

        #yij
        for i in range(n_customers):
            for j in range(n_facilities):
                var = []
                var.append(cluster_dict['second_stage']["q_c_f"][s][i][j])
                var.append(cluster_dict['second_stage']["h"][s][i])
                cflp_feature.append(var)
        
        #zj
        for j in range(n_facilities):
            var = []
            var.append(cluster_dict["second_stage"]["recourse_costs"])
            var.append(cluster_dict["second_stage"]['z_coeff'])
            cflp_feature.append(var)
     
        #生成场景的特征以及adj
        features = np.array(cflp_feature)
        features = lil_matrix(features.astype(float))
        #features, _ = preprocess_features(features)
        features = features.todense()
        features = torch.FloatTensor(features[np.newaxis])
        if all_feature == None:
            all_feature = features
        else :
            all_feature = torch.cat((all_feature, features), dim = 0)
    
    return adjs, all_feature, cons_feature


#生成batch数据  --------------------------
def padding_feat(feat):
    a,b,c = feat.shape
    
    # Desired size [200, 220, 2]
    desired_size = (a, b, 2)

    # Create a new tensor of the desired size filled with zeros
    new_tensor = torch.zeros(desired_size)

    # Copy the original tensor into the new tensor
    # The original tensor is copied into the top left corner of the new tensor
    new_tensor[:a, :b,:1] = feat
    return new_tensor

def build_batch_data(all_adjs,all_var_feats,all_con_feats, all_edge_scen):
    batch_datas = []
    for i in tqdm(range(len(all_adjs)), desc="处理batch"):
        adjs = all_adjs[i]
        var_feats = all_var_feats[i]
        con_feats = padding_feat(all_con_feats[i])
        feats = torch.cat((con_feats,var_feats),dim=1)
        mark_node = torch.zeros(feats.shape[1],)
        mark_node[:con_feats.shape[1]] = 1
        edge_indices = [torch.stack([torch.from_numpy(adjs[i].row), torch.from_numpy(adjs[i].col)], dim=0) for i in range(len(adjs))]
        values = [torch.from_numpy(adjs[i].data) for i in range(len(adjs))]
        datas = [Data(x=feats[i], edge_index= edge_indices[i], edge_attr= values[i], mark=mark_node) for i in range(len(adjs))]
        batched_data = Batch.from_data_list(datas)
        graph = Data(x=batched_data.x, edge_index=batched_data.edge_index, edge_attr=batched_data.edge_attr, batch=batched_data.batch, scen_adj= all_edge_scen[i], mark=batched_data.mark, ptr= batched_data.ptr )

        batch_datas.append(graph)
    return batch_datas


#处理特征为batch图数据  --------------------------
def process_cluster_file(cluster_file, cluster_dict, feature_type, mp_data_list):
    print("Now:", cluster_file)
    if feature_type == 'cflp':
        adjs, all_feature, cons_feature = _get_cflp_features(cluster_dict)
        edge_adj = torch.from_numpy(cluster_dict['second_stage']["h"]).float()
        edge_adj = F.cosine_similarity(edge_adj.unsqueeze(1), edge_adj.unsqueeze(0), dim=2)
    print("!!Done:", cluster_file)
    mp_data_list.append((cluster_file, adjs, all_feature, cons_feature, edge_adj))

def process_data(source_folder, cluster_dicts, feature_type, process_type):
    if process_type == "train":
        print(f"{process_type}...")
        all_adjs, all_var_feats, all_con_feats, all_edge_scen, cls = [], [], [], [], []
        start = time.time()

        # 设置使用的进程数量
        num_processes = 2  # 根据实际情况设置进程数量

        with Manager() as manager:
            mp_data_list = manager.list()
            pool = Pool(num_processes)

            for cluster_file in tqdm(cluster_dicts, desc="处理进度", unit="文件"):
                orin_file = cluster_file
                cluster_file = os.path.join(source_folder, cluster_file)
                with open(cluster_file, "rb") as f:
                    cluster_dict = pickle.load(f)
                pool.apply_async(process_cluster_file, args=(orin_file, cluster_dict, feature_type, mp_data_list))

            pool.close()  # 关闭进程池
            pool.join()  # 等待所有任务完成

            data = list(mp_data_list)

        for l in range(len(data)):
            cluster_file, adjs, all_feature, cons_feature, edge_adj = data[l]
            cls.append(cluster_file)
            all_edge_scen.append(edge_adj)
            all_adjs.append(adjs)
            all_var_feats.append(all_feature)
            all_con_feats.append(cons_feature)
    
        print("used time:", time.time() - start)

    
        batch_graphs = build_batch_data(all_adjs,all_var_feats,all_con_feats,all_edge_scen)

        total_size = len(batch_graphs)  

        # change !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        train_size =  8192


        # 创建随机的索引列表
        index_list = list(range(total_size))
        random.shuffle(index_list)

        # 划分训练集和测试集的索引
        train_index = index_list[:train_size]
        eval_index = index_list[train_size:]

        train_dataset = [batch_graphs[i] for i in train_index]
        train_cls = [cls[i] for i in train_index]
        eval_dataset =  [batch_graphs[i] for i in eval_index]
        eval_cls = [cls[i] for i in eval_index]

        return train_dataset, train_cls, eval_dataset, eval_cls

    else:
        print(f"{process_type}...")
        all_adjs, all_var_feats, all_con_feats, all_edge_scen, cls = [], [], [], [], []
        start = time.time()
        # 设置使用的进程数量
        num_processes = 2  # 根据实际情况设置进程数量

        with Manager() as manager:
            mp_data_list = manager.list()
            pool = Pool(num_processes)

            for cluster_file in tqdm(cluster_dicts, desc="处理进度", unit="文件"):
                orin_file = cluster_file
                cluster_file = os.path.join(source_folder, cluster_file)
                with open(cluster_file, "rb") as f:
                    cluster_dict = pickle.load(f)
                pool.apply_async(process_cluster_file, args=(orin_file, cluster_dict, feature_type, mp_data_list))

            pool.close()  # 关闭进程池
            pool.join()  # 等待所有任务完成

            data = list(mp_data_list)

        for l in range(len(data)):
            cluster_file, adjs, all_feature, cons_feature, edge_adj = data[l]
            cls.append(cluster_file)
            all_edge_scen.append(edge_adj)
            all_adjs.append(adjs)
            all_var_feats.append(all_feature)
            all_con_feats.append(cons_feature)
    
        print("used time:", time.time() - start)
        batch_graphs = build_batch_data(all_adjs,all_var_feats,all_con_feats,all_edge_scen)
        total_size = len(batch_graphs)  
        index_list = list(range(total_size))
        test_index = index_list
        test_dataset =  [batch_graphs[i] for i in test_index]
        test_cls = [cls[i] for i in test_index]

        return test_dataset, test_cls, 0, 0




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gnn_Transformer for two_stage")
    parser.add_argument('--config_file', type=str, default='./configs/cflp_config.json', help="base config json dir")
    parser.add_argument('--scenarios_folder', type=str, default='./train_scenarios/', help="base scenarios folder")
    parser.add_argument('--feature_type', type=str, default='cflp', help="type of problem")
    parser.add_argument('--process_type', type=str, default='train', help="type of process")
    args = parser.parse_args()
    source_folder = args.scenarios_folder
    all_kwargs = json.load(open(args.config_file, 'r'))
    if args.process_type == 'train':
        train_data_param = all_kwargs['TrainData']
    else:
        test_data_param = all_kwargs['TestData']
    print("TYPE:", args.feature_type)

    cluster_dicts = sorted(os.listdir(source_folder), key=sort_key)

#-------------------------------------- 选择多少个数据 !!!!!!!!!
    k = min(8292, len(cluster_dicts))                                              #前k个

    cluster_dicts = cluster_dicts[:k]
#--------------------------------------
    
    #存储数据集
    if args.process_type == 'train':
        #train
        train_save_path = train_data_param['save_path']                               #训练集特征的位置
        train_cls_path = train_data_param['cls_path']                                 #训练集场景字典的位置

        #eval
        eval_save_path = os.path.join(all_kwargs['train']['eval_path'], all_kwargs['train']['eval_pt'])  #验证集特征的位置
        eval_cls_path = os.path.join(all_kwargs['train']['eval_path'], all_kwargs['train']['eval_cls'])  #验证集场景字典的位置
        train_dataset, train_cls, eval_dataset, eval_cls= process_data(source_folder, cluster_dicts, feature_type=args.feature_type, process_type=args.process_type)
        torch.save(train_dataset, train_save_path)
        torch.save(eval_dataset, eval_save_path)
        with open(train_cls_path, 'wb') as f:
            pickle.dump(train_cls, f)
        with open(eval_cls_path, 'wb') as f:
            pickle.dump(eval_cls, f)

        print("!!Done Process Train Data!!")
    else:
        #test
        test_save_path = test_data_param['save_path']                                 #测试集特征的位置
        test_cls_path = test_data_param['cls_path']                                   #测试集场景字典的位置
        test_dataset, test_cls, _, _ = process_data(source_folder, cluster_dicts, feature_type=args.feature_type, process_type=args.process_type)
        torch.save(test_dataset, test_save_path)

        with open(test_cls_path, 'wb') as f:
            pickle.dump(test_cls, f)
        print("!!Done Process Test Data!!")