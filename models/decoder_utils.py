import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os  #控制程序结束 可以定位查看结果 os._exit(0)


class Env():
	def __init__(self, device, node_embeddings):
		super().__init__()
		"""
			node_embeddings: (batch, n_nodes, embed_dim)
			h1:(batch, 1, embed_dim)
			Nodes that have been visited will be marked with True.
		"""
		self.device = device
		self.node_embeddings = node_embeddings
		self.batch, self.n_nodes, self.embed_dim = node_embeddings.size()
		self.h1 = None  #context中不变的部分
		self.visited_scenarios = torch.zeros((self.batch, self.n_nodes, 1), dtype = torch.bool).to(self.device)

	def _create_t1(self):
		mask_t1 = self.visited_scenarios.to(self.device)
		return mask_t1
	
	def _get_step(self, next_node):
		one_hot = torch.eye(self.n_nodes)[next_node.cpu()]		
		visited_mask = one_hot.type(torch.bool).permute(0,2,1).to(self.device)
		self.visited_scenarios = self.visited_scenarios | visited_mask
		mask_t1 = self.visited_scenarios.to(self.device)
		prev_node_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = next_node[:,:,None].repeat(1,1,self.embed_dim))
		if self.h1 == None:
			self.h1 = prev_node_embedding
		else :
			self.h1 = (self.h1 + prev_node_embedding) / 2  
		step_context = torch.cat([prev_node_embedding, self.h1], dim = -1).to(self.device)
		return mask_t1, step_context
	
	def get_log_likelihood(self, _log_p, pi):
		""" _log_p: (batch, decode_step, n_nodes)
			pi: (batch, decode_step)
		"""
		log_p = torch.gather(input = _log_p, dim = 2, index = pi[:,:,None])  #gather中dim等于k就在这个维度上每次寻找相应值
		return torch.sum(log_p.squeeze(-1), 1)
	
	def get_entropy(self, _log_p, pi):
		"""_log_p: (batch, decode_step, n_nodes)
            pi: (batch, decode_step)
        """
		log_p = torch.gather(input=_log_p, dim=2, index=pi[:, :, None])
		prob = torch.exp(log_p.squeeze(-1))
		entropy = torch.sum(-prob * log_p.squeeze(-1), dim=1)
		return entropy


#选点器
class Sampler(nn.Module):
	""" args; logits: (batch, n_nodes)
		return; next_node: (batch, 1)
		TopKSampler <=> greedy; sample one with biggest probability
		CategoricalSampler <=> sampling; randomly sample one from possible distribution based on probability
	"""
	def __init__(self, n_samples = 1, **kwargs):
		super().__init__(**kwargs)
		self.n_samples = n_samples
		
class TopKSampler(Sampler):
	def forward(self, logits):
		return torch.topk(logits, self.n_samples, dim = 1)[1]# == torch.argmax(log_p, dim = 1).unsqueeze(-1)

class CategoricalSampler(Sampler):
	def forward(self, logits):
		return torch.multinomial(logits.exp(), self.n_samples)