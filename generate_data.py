import os, sys
import pickle
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import gurobipy as gp

# from stochoptim.stochprob.facility_location.facility_location_problem  and add "demands"
def generate_random_parameters(n_facility_locations, n_client_locations, n_zones, seed=None):
    """Generate randomly a set of deterministic parameters of the facility location problem"""
    if seed is not None:
        np.random.seed(seed)
    
    
    return {"pos_client": np.random.rand(n_client_locations, 2),
            "pos_facility": np.random.rand(n_facility_locations, 2),
            "opening_cost": np.random.uniform(600, 1500, n_facility_locations),
            "facility_capacity": np.random.uniform(100, 150, n_facility_locations),
            "max_facilities": 8,
            "min_facilities_in_zone": np.array([1] * n_zones),
            "facility_in_zone": np.random.choice(range(n_zones), size=n_facility_locations),
            "penalty": 1000 * np.ones(n_facility_locations),
            "demands": np.random.randint(5, 105 + 1, size=n_client_locations)}


def solve_cflp_relax(cluster_dict, idx):
    """Formulates two stage extensive form and solves"""
    n_customers = cluster_dict["n_customers"]
    n_facilities = cluster_dict["n_facilities"]
    n_scenarios = np.random.choice(range(cluster_dict['n_scenarios']), size=100, replace=False)

    prob = 1.0/len(n_scenarios)
    model = gp.Model()
    var_dict = {}

    # 目标函数
    # 每个设施的二进制变量 第一阶段
    for i in range(n_facilities):
        var_name = f"x_{i}"
        var_dict[var_name] = model.addVar(
            lb=0.0,
            ub=1.0,
            obj=cluster_dict["first_stage"]["facilities_cost"][i],
            vtype="B",
            name=var_name,
        )
    #第一阶段约束 设施数不大于v
    cons = 0
    for i in range(n_facilities):
        cons += var_dict[f"x_{i}"]
    model.addConstr(cons<=cluster_dict['first_stage']['constraint'], name = f"v")

    for s in n_scenarios:
        # 第二阶段
        for i in range(n_customers):
            for j in range(n_facilities):
                var_name = f"y_{i}_{j}_{s}"
                var_dict[var_name] = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    obj=prob * cluster_dict["second_stage"]["trans_cost"][i][j],
                    vtype="B",
                    name=var_name,
                )
        for j in range(n_facilities):
            var_name = f"z_{j}_{s}"
            var_dict[var_name] = model.addVar(
                lb=0.0,
                obj=prob * cluster_dict["second_stage"]["recourse_costs"],
                vtype="C",
                name=var_name,
            )

        # 约束条件
        #约束1
        for j in range(n_facilities):
            cons = (-1.0)*cluster_dict['second_stage']["qx"][j] * var_dict[f"x_{j}"] - var_dict[f"z_{j}_{s}"]*cluster_dict['second_stage']['z_coeff'] 
            for i in range(n_customers):
                cons+= cluster_dict['second_stage']["q_c_f"][s][i][j]*var_dict[f"y_{i}_{j}_{s}"]
            model.addConstr(cons<=0, name = f"c_{j}_{s}")
        #约束2
        for j in range(n_facilities):
            cons = var_dict[f"z_{j}_{s}"] - cluster_dict['second_stage']['M'] * var_dict[f"x_{j}"] 
            model.addConstr(cons<=0, name = f"d_{j}_{s}")
        #约束3
        for i in range(n_customers):
            cons = (-1)*cluster_dict['second_stage']["h"][s][i]
            for j in range(n_facilities):
                cons+=var_dict[f"y_{i}_{j}_{s}"]
            model.addConstr(cons==0, name = f"t_{i}_{s}")
    
    model.update()

    #set param
    model.setParam('OutputFlag', 0) 
    model.setParam("MIPGap", 0.000)
    model.setParam("TimeLimit", 10800)

    # 设置线程数
    model.setParam('Threads', 4)

    model.optimize()

    # 获取第一阶段变量的取值 
    variable_values = {var.varName: var.x for var in model.getVars() if 'x' in var.varName}

    solving_results = {}
    solving_results["primal"] = model.objVal
    solving_results["time"] = model.Runtime
    solving_results['scenarios'] = n_scenarios
    solving_results['index'] = idx
    solving_results['X'] = variable_values

    #print("solving_results:",solving_results)
    return solving_results



def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # first-stage dataset
    n_facility_locations = 10
    n_client_locations = 20
    n_zones = 1
    #------------
    client_node = []
    facility_node = [] 
    dist = []
    n_scenarios = 200
    p = 0.8
    n_procs = 3
    #------------------
    scenarios = [] 
    all_tran_cost = []
    hs = []
    demands = []
    z_coeffs = []
    recourse_costs = []
    fac_costs = []
    fac_capacity = []
    cls = []

    for i in tqdm(range(args.num_total_ins)):
        param = generate_random_parameters(n_facility_locations, n_client_locations, n_zones)
        pos_c = param['pos_client']
        pos_f = param['pos_facility']
        
        fac_costs.append(param['opening_cost'])
        fac_capacity.append(param['facility_capacity'])

        open_cost = param['opening_cost']/1500.
        fac_cap = param['facility_capacity']/150. 

        z_coeff = np.ones(n_facility_locations)
        recourse_cost = np.ones(n_facility_locations)*1000
        z_coeffs.append(z_coeff)
        recourse_costs.append(recourse_cost)

        facility_node.append(np.concatenate((pos_f,open_cost[:,None],fac_cap[:,None]),axis=1)) ### n_facility_locations*4
        client_node.append(pos_c)   
        tran_cost =  np.sqrt(
            (pos_c[:,0].reshape((-1, 1)) - pos_f[:,0].reshape((1, -1))) ** 2
            + (pos_c[:,1].reshape((-1, 1)) - pos_f[:,1].reshape((1, -1))) ** 2) \
                              * 10 * param['demands'].reshape((-1, 1))
        all_tran_cost.append(tran_cost)        
        customer_presence = np.random.binomial(1, np.random.uniform(0.8, 
                                                               0.9, 
                                                               n_client_locations), 
                                          (n_scenarios, n_client_locations))
        hs.append(customer_presence)
        q_c_f = np.random.uniform(20, 80, size=(n_scenarios, n_client_locations, n_facility_locations))
        demands.append(q_c_f)

        #创建字典 并存储
        cluster_dict={}
        cluster_dict['n_customers']=n_client_locations
        cluster_dict['n_facilities']=n_facility_locations
        cluster_dict['n_scenarios']=n_scenarios
        cluster_dict['first_stage']={}
        cluster_dict['first_stage']['facilities_cost']= param['opening_cost']  #对应of
        cluster_dict['first_stage']['constraint']=8   #v的值
        cluster_dict['second_stage']={}
        cluster_dict['second_stage']['recourse_costs'] = 1000  #bf
        cluster_dict['second_stage']["q_c_f"] = q_c_f #qcf
        cluster_dict['second_stage']["qx"]= param['facility_capacity']
        cluster_dict['second_stage']['M']= 1e4
        cluster_dict['second_stage']["h"]= customer_presence 
        cluster_dict['second_stage']['trans_cost']  = tran_cost
        cluster_dict['second_stage']['z_coeff'] = 1
        cluster_dict['pos_c'] = pos_c
        cluster_dict['pos_f'] = pos_f
        file_path = os.path.join(args.file_path, f"scene_200_{i}.pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(cluster_dict, file)
        
        # 求解结果
        results = solve_cflp_relax(cluster_dict, i)
        file_path = os.path.join(args.result_path, f"result_of_{i}.pkl")
        with open(file_path, 'wb') as rf:
            pickle.dump(results, rf)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_total_ins", type=int, default=10)  
    parser.add_argument("--file_path", type=str, default="./train_scenarios")
    parser.add_argument("--result_path", type=str, default="./train_results")
    args = parser.parse_args()
    main(args)
