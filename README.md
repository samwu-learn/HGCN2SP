# HGCN2SP: Hierarchical Graph Convolutional Network for Two-Stage Stochastic Programming
Author: Yang Wu, Yifan Zhang, Zhenxing Liang, Jian Cheng

## 项目概述

HGCN2SP是一个创新的框架，结合了层次化图卷积网络和强化学习方法，用于解决两阶段随机规划问题。本项目聚焦于CFLP，通过学习场景特征，实现高效的求解。


## 环境需求

- Python 3.8+
- PyTorch 1.10+
- PyG (PyTorch Geometric)
- Gurobi
- NumPy, tqdm, wandb (可选)

## 项目结构

```
HGCN2SP/
├── configs              # 配置文件目录
├── data                 # 存储训练和测试数据
├── eval_instance        # 存储验证数据
├── model_path           # 存储模型权重
├── models               # 网络模型组件
├── test_csv             # 存储测试csv
├── train/test_scenarios # 存储训练/测试实例
├── train/test_results   # 存储训练/测试标准结果
├── utils                # 工具库
├── agent.py             # 智能体模型定义
├── env.py               # 环境交互接口
├── sample.py            # 数据采样器
├── trainer.py           # PPO训练器
├── generate_data.py     # 数据生成脚本
├── process_data_new.py  # 数据预处理脚本
├── run.py               # 主运行脚本

```

## 运行流程

### 第一步：生成问题实例数据

首先需要生成CFLP问题实例及其最优解：

```bash
python generate_data.py --seed 0 --num_total_ins 100 --file_path ./train_scenarios --result_path ./train_results
```

参数说明：
- `--seed`: 随机种子
- `--num_total_ins`: 生成的问题实例数量
- `--file_path`: 问题实例存储路径
- `--result_path`: 求解结果存储路径

### 第二步：数据预处理

处理原始数据，创建数据集：

```bash
python process_data_new.py
```

该脚本会根据配置文件中指定的路径读取问题实例和求解结果，构建适合模型输入的数据结构。使用时请修改里面的参数以免报错。

### 第三步：模型训练与评估

运行主脚本进行模型训练/测试：

```bash
python run.py --config_file ./configs/cflp_config.json 
```

参数说明：
- `--config_file`: 配置文件路径

训练时请将Args中的mode修改为"train"，测试时修改为"test"，并更新model_test_path字段


## 配置文件说明

配置文件`cflp_config.json`包含以下主要部分：

1. **Policy**: 控制策略网络结构
   - `var_dim`, `con_dim`: 变量和约束特征维度
   - `l_hid_dim`, `h_hid_dim`: 局部和全局隐藏层维度
   - `n_heads`: 注意力头数量

2. **TrainData/TestData**: 数据配置
   - `n_scenarios`: 场景数量
   - `pkl_folder`, `result_folder`: 数据和结果路径
   - `save_path`, `cls_path`: 处理后数据保存路径

3. **train**: 训练配置
   - `sel_num`: 选择的场景数量
   - `decode_type`: 解码策略类型
   - `eval_cls_loc`: 验证数据路径
   - `eval_result`: 验证数据结果路径
   - `eval_epoch`: 评估频率

4. **test**: 测试配置
   - `sel_num`: 测试时选择的场景数量
   - `decode_type`: 测试解码策略

修改配置文件可以调整网络结构、训练参数和数据路径等设置，请根据实际路径修改配置。

## 引用

如果您在研究中使用了HGCN2SP，请引用我们的论文：

```bibtex
@article{wu2024hgcn2sp,
  title={HGCN2SP: Hierarchical Graph Convolutional Network for Two-Stage Stochastic Programming},
  author={Wu, Yang and Zhang, Yifan and Liang, Zhenxing and Cheng, Jian},
  journal={arXiv preprint},
  year={2024}
}
```


