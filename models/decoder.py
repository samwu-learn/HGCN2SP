import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
# from . import layers
from .layers import MultiHeadAttention, DotProductAttention
from .decoder_utils import TopKSampler, CategoricalSampler, Env
import os


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

class ValueCell(nn.Module):
	def __init__(self, embed_dim = 64,**kwargs):
		super().__init__(**kwargs)
		self.embed_dim = embed_dim
		self.Wv = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wk = nn.Linear(embed_dim, embed_dim, bias = False)
		self.MLP = nn.Sequential(
            self.layer_init(nn.Linear(embed_dim, 128)),      #原始数据是32
            nn.Tanh(),
            self.layer_init(nn.Linear(128, 256)),
            nn.Tanh(),
            self.layer_init(nn.Linear(256, 1), std=1.0),
        )
		self.scale = math.sqrt(embed_dim)
		self.inint_weight()
	
	def inint_weight(self):
		orthogonal_init(self.Wv)
		orthogonal_init(self.Wq)
		orthogonal_init(self.Wk)

	def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
		torch.nn.init.orthogonal_(layer.weight, std)
		torch.nn.init.constant_(layer.bias, bias_const)
		return layer

	
	def forward(self, graph, x):
		Q = self.Wq(graph[:, None, :])
		K = self.Wk(x)
		V = self.Wv(x)
		logits = torch.matmul(Q, K.transpose(-1,-2)) / self.scale
		attn_weights = nn.functional.softmax(logits , dim=-1)
		attended_values = torch.matmul(attn_weights, V).squeeze(dim = 1)
		value = self.MLP(attended_values) 
		return value

class CriticCell(nn.Module):
	def __init__(self, embed_dim = 64, is_init = True,**kwargs):
		super().__init__(**kwargs)
		self.embed_dim = embed_dim
		self.Wv = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wk = nn.Linear(embed_dim, embed_dim, bias = False)
		self.MLP = nn.Sequential(
            self.layer_init(nn.Linear(embed_dim, 128)),      #原始数据是32
            nn.Tanh(),
			nn.Dropout(p=0.5),
            self.layer_init(nn.Linear(128, 256)),
            nn.Tanh(),
            self.layer_init(nn.Linear(256, 1), std=1.0),
        )
		self.scale = math.sqrt(embed_dim)
		if is_init:
			self.init_parameters()

	
	def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
		torch.nn.init.orthogonal_(layer.weight, gain=1.0)
		torch.nn.init.constant_(layer.bias, bias_const)
		return layer
	
	def init_parameters(self):     #初始化参数
		orthogonal_init(self.Wv)
		orthogonal_init(self.Wq)
		orthogonal_init(self.Wk)

	
	def forward(self, graph, x):
		Q = self.Wq(graph[:, None, :])
		K = self.Wk(x)
		V = self.Wv(x)
		logits = torch.matmul(Q, K.transpose(-1,-2)) / self.scale
		attn_weights = nn.functional.softmax(logits , dim=-1)
		attended_values = torch.matmul(attn_weights, V).squeeze(dim = 1)
		value = self.MLP(attended_values)
		return value

class DecoderCell(nn.Module):
	def __init__(self, embed_dim = 64, n_heads = 8, clip = 10., **kwargs):
		super().__init__(**kwargs)

		self.embed_dim = embed_dim
		
		self.Wk1 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wv = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wk2 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias = False)  #不变的全局信息
		self.Wout = nn.Linear(embed_dim, embed_dim, bias = False)      #单头注意力的Q
		self.Wq_step = nn.Linear(2*embed_dim, embed_dim, bias = False) #根据实际情况更改conext
		self.W_placeholder = nn.Parameter(torch.Tensor(2 * embed_dim)) #初始t=1的context变量
		self.W_placeholder.data.uniform_(-1, 1)
		

		self.MHA = MultiHeadAttention(n_heads = n_heads, embed_dim = embed_dim, need_W = False)
		self.SHA = DotProductAttention(clip = clip, return_logits = True, head_depth = embed_dim)
		self.env = Env
		self.init_parameters()

	def init_parameters(self):
		orthogonal_init(self.Wk1)
		orthogonal_init(self.Wv)
		orthogonal_init(self.Wk2)
		orthogonal_init(self.Wq_fixed)
		orthogonal_init(self.Wq_step)
		orthogonal_init(self.Wout, gain=0.01)

			
	def compute_static(self, node_embeddings, graph_embedding): #固定不变的参数
		self.Q_fixed = self.Wq_fixed(graph_embedding[:,None,:])  #多头的Q
		self.K1 = self.Wk1(node_embeddings)
		self.V = self.Wv(node_embeddings)      #多头的K和V
		self.K2 = self.Wk2(node_embeddings)    #单头的K
		
	def compute_dynamic(self, mask, step_context):
		Q_step = self.Wq_step(step_context)
		Q1 = self.Q_fixed + Q_step
		Q2 = self.MHA([Q1, self.K1, self.V], mask = mask)
		Q2 = self.Wout(Q2)
		logits = self.SHA([Q2, self.K2, None], mask = mask) 
		return logits.squeeze(dim = 1) #-->输出为(batch, n_nodes)

	def forward(self, device, encoder_output, return_cost = True, cluster_k = 3, decode_type = 'sampling', cls = None, action = None):
		node_embeddings, graph_embedding = encoder_output
		self.compute_static(node_embeddings, graph_embedding)
		env = Env(device, node_embeddings)
		mask = env._create_t1()

		step_context = self.W_placeholder[None, None, :].expand(env.batch, 1, self.W_placeholder.size(-1))

		selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
		log_ps, tours = [], []	

		for i in range(cluster_k):
			logits = self.compute_dynamic(mask, step_context)
			log_p = torch.log_softmax(logits, dim = -1)
			next_node = selecter(log_p)
			mask, step_context= env._get_step(next_node)
			tours.append(next_node.squeeze(1))  #选择的场景
			log_ps.append(log_p)                #概率
			if env.visited_scenarios.all():
				break

		pi = torch.stack(tours, 1)
		if action is not None:
			pi = action

		
		ll = env.get_log_likelihood(torch.stack(log_ps, 1), pi)
		entropy = env.get_entropy(torch.stack(log_ps, 1), pi)
		    
		
		if return_cost:
			log_p = torch.gather(input = torch.stack(log_ps, 1), dim = 2, index = pi[:,:,None])
			return pi, log_p
		return pi, ll, entropy