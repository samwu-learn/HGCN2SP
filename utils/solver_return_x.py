from itertools import product
from multiprocessing import Manager, Pool

from gurobipy import GRB
import gurobipy as gp
import numpy as np
import torch

def solve_cflp_scenarios(cluster, scenarios = None, weights = None, first_precision = None, flag=False, mip_gap=0, large = False):
    
    constraint = cluster['first_stage']['constraint']
    M = cluster['second_stage']['M'] 
    model = gp.Model()
    var_dict = {}

    n_facilities = cluster['n_facilities']
    n_customers = cluster['n_customers']
    if scenarios is not None:
        index = scenarios
    else:
        index = list(range(cluster['n_scenarios']))
    if weights is None:
        prob = 1.0 / len(index)
        weights = [prob for i in range(len(index))]

    # 目标函数
    # 每个设施的二进制变量 第一阶段
    for i in range(n_facilities):
        var_name = f"x_{i}"
        if first_precision is None:
            var_dict[var_name] = model.addVar(
                lb=0.0,
                ub=1.0,
                obj=cluster["first_stage"]["facilities_cost"][i],
                vtype="B",
                name=var_name,
            )
        else :
            var_dict[var_name] = model.addVar(   #固定第一阶段决策变量
                lb=first_precision[f"x_{i}"],
                ub=first_precision[f"x_{i}"],
                obj=cluster["first_stage"]["facilities_cost"][i],
                vtype="B",
                name=var_name,
            )
    #第一阶段约束 设施数不大于v
    cons = 0
    for i in range(n_facilities):
        cons += var_dict[f"x_{i}"]
    model.addConstr(cons<=constraint, name = f"v")

    loc = 0
    for s in index:
        # 第二阶段
        for i in range(n_customers):
            for j in range(n_facilities):
                var_name = f"y_{i}_{j}_{s}"
                var_dict[var_name] = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    obj=weights[loc] * cluster["second_stage"]["trans_cost"][i][j],
                    vtype="B",
                    name=var_name,
                )
        for j in range(n_facilities):
            var_name = f"z_{j}_{s}"
            var_dict[var_name] = model.addVar(
                lb=0.0,
                obj=weights[loc] * cluster["second_stage"]["recourse_costs"],
                vtype="C",
                name=var_name,
            )
        loc = loc + 1

        # 约束条件
        #约束1
        for j in range(n_facilities):
            cons = (-1.0)*cluster['second_stage']["qx"][j] * var_dict[f"x_{j}"] - var_dict[f"z_{j}_{s}"]*cluster['second_stage']['z_coeff'] 
            for i in range(n_customers):
                cons+= cluster['second_stage']["q_c_f"][s][i][j]*var_dict[f"y_{i}_{j}_{s}"]
            model.addConstr(cons<=0, name = f"c_{j}_{s}")
        #约束2
        for j in range(n_facilities):
            cons = var_dict[f"z_{j}_{s}"] - cluster['second_stage']['M'] * var_dict[f"x_{j}"] 
            model.addConstr(cons<=0, name = f"d_{j}_{s}")
        #约束3
        for i in range(n_customers):
            cons = (-1)*cluster['second_stage']["h"][s][i]
            for j in range(n_facilities):
                cons+=var_dict[f"y_{i}_{j}_{s}"]
            model.addConstr(cons==0, name = f"t_{i}_{s}")
    
    model.update()

    #set param
    if flag == True:
        model.setParam('outputFlag',1)
    else:
        model.setParam('OutputFlag', 0) 
    
    if large:
        model.setParam("MIPGap", 0.05)
    else:
        model.setParam("MIPGap", mip_gap)
    model.setParam("TimeLimit", 10800)
    # 设置线程数
    model.setParam('Threads', 16)

    model.optimize()

    solving_results = {}

    if model.status == GRB.OPTIMAL:
        # 获取第一阶段变量的取值 
        if first_precision == None:
            variable_values = {var.varName: var.x for var in model.getVars() if 'x' in var.varName}
        else :
            variable_values = first_precision
        solving_results["primal"] = model.objVal
        solving_results["time"] = model.Runtime
        solving_results['X'] = variable_values
    
    elif model.status == GRB.INFEASIBLE or model.status == GRB.UNBOUNDED:
        solving_results["primal"] = 1
        solving_results["time"] = 100
        solving_results['X'] = first_precision

    else :
        if first_precision == None:
            variable_values = {var.varName: var.x for var in model.getVars() if 'x' in var.varName}
        else :
            variable_values = first_precision
        solving_results["primal"] = model.objVal
        solving_results["time"] = model.Runtime
        solving_results['X'] = variable_values

    return solving_results

def solve_cflp_softmax(args):   
    """
       args: (dict, list, bool)
    """
    
    cluster_dict, index, is_train = args
    """Formulates two stage extensive form and solves"""
    if is_train:
        index = torch.squeeze(index).numpy()
        
        train_results = solve_cflp_scenarios(cluster_dict, index)
        train_results = solve_cflp_scenarios(cluster_dict, first_precision=train_results['X'])
        results = [train_results['primal'], train_results['time']]
        results =  torch.tensor(results).unsqueeze(0)
        return results
    n_customers = cluster_dict["n_customers"]
    n_facilities = cluster_dict["n_facilities"]
    
    n_scenarios = index
    
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

    model.setParam('OutputFlag', 0) 
    model.setParam("MIPGap", 0.001)
    model.setParam("TimeLimit", 600)
    
    # 设置线程数
    model.setParam('Threads', 16)

    model.optimize()

    solving_results = {}

    solving_results["primal"] = model.objVal
    solving_results["time"] = model.Runtime

    return solving_results

def solve_cflp_softmax_new(args):   
    """
       args: (dict, list, bool)
    """
    if len(args) ==3:
        cluster_dict, index, is_train = args
        is_large = False
    else:
        cluster_dict, index, is_train, is_large = args
    """Formulates two stage extensive form and solves"""
    if is_train:
        index = torch.squeeze(index).numpy()
        #print("index:",index)
        if len(index) < 20 :
            mip_gap = 0
        else:
            mip_gap = 0.005
        train_results = solve_cflp_scenarios(cluster_dict, index, flag=False, mip_gap=mip_gap,large=is_large)
        x = []
        X = train_results['X']
        for key in X.keys():
            x.append(X[key])
        x.append(train_results['time'])
        train_results = solve_cflp_scenarios(cluster_dict, first_precision=train_results['X'])
        x.append(train_results['primal'])
        results =  torch.tensor(x)
        return results
    n_customers = cluster_dict["n_customers"]
    n_facilities = cluster_dict["n_facilities"]
    
    n_scenarios = index
    
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

    model.setParam('OutputFlag', 0) 
    model.setParam("MIPGap", 0.001)
    model.setParam("TimeLimit", 600)
    
    # 设置线程数
    model.setParam('Threads', 4)

    model.optimize()

    solving_results = {}

    solving_results["primal"] = model.objVal
    solving_results["time"] = model.Runtime

    return solving_results


