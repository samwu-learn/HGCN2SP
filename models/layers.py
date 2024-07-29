import torch
import torch.nn as nn
import math

class DotProductAttention(nn.Module):
	def __init__(self, clip = None, return_logits = False, head_depth = 16, inf = 1e+10, **kwargs):
		super().__init__(**kwargs)
		self.clip = clip
		self.return_logits = return_logits
		self.inf = inf
		self.scale = math.sqrt(head_depth)

	def forward(self, x, mask = None):
		""" Q: (batch, n_heads, q_seq(=n_nodes or =1), head_depth) 
			K: (batch, n_heads, k_seq(=n_nodes), head_depth)
			logits: (batch, n_heads, q_seq(this could be 1), k_seq)
			mask: (batch, n_nodes, 1), e.g. tf.Tensor([[ True], [ True], [False]])
			mask[:,None,None,:,0]: (batch, 1, 1, n_nodes) ==> broadcast depending on logits shape
			[True] -> [1 * -np.inf], [False] -> [logits]
			K.transpose(-1,-2).size() == K.permute(0,1,-1,-2).size()
		"""
		Q, K, V = x
		logits = torch.matmul(Q, K.transpose(-1,-2)) / self.scale

		if self.clip is not None:
			logits = self.clip * torch.tanh(logits)
			
		if self.return_logits:
			if mask is not None:
				return logits.masked_fill(mask.permute(0,2,1) == True, -self.inf)
			return logits

		if mask is not None:
			logits = logits.masked_fill(mask[:,None,None,:,0].repeat(1,logits.size(1),1,1) == True, -self.inf)
			
		probs = torch.softmax(logits, dim = -1)

		return torch.matmul(probs, V)

class MultiHeadAttention(nn.Module):
	def __init__(self, n_heads = 8, embed_dim = 128, clip = None, return_logits = None, need_W = None):
		super().__init__()
		self.n_heads = n_heads
		self.embed_dim = embed_dim
		self.head_depth = self.embed_dim // self.n_heads
		if self.embed_dim % self.n_heads != 0:
			raise ValueError("embed_dim = n_heads * head_depth")
		
		self.need_W = need_W 
		self.attention = DotProductAttention(clip = clip, return_logits = return_logits, head_depth = self.head_depth)
		if self.need_W:
			self.Wk = nn.Linear(embed_dim, embed_dim, bias = False)
			self.Wv = nn.Linear(embed_dim, embed_dim, bias = False)
			self.Wq = nn.Linear(embed_dim, embed_dim, bias = False)
			self.Wout = nn.Linear(embed_dim, embed_dim, bias = False)
		self.init_parameters()
	
	def init_parameters(self):     #初始化参数
		for name, param in self.named_parameters():
			if name == 'Wout.weight':
				stdv = 1. / math.sqrt(param.size(-1))
			elif name in ['Wk.weight', 'Wv.weight', 'Wq.weight']:
				stdv = 1. / math.sqrt(self.head_depth)
			else:
				raise ValueError
			param.data.uniform_(-stdv, stdv)

	def split_heads(self, T):
		""" https://qiita.com/halhorn/items/c91497522be27bde17ce
			T: (batch, n_nodes, self.embed_dim)
			T reshaped: (batch, n_nodes, self.n_heads, self.head_depth)
			return: (batch, self.n_heads, n_nodes, self.head_depth)
			
			https://raishi12.hatenablog.com/entry/2020/04/20/221905
		"""
		shape = T.size()[:-1] + (self.n_heads, self.head_depth)
		T = T.view(*shape)
		return T.permute(0,2,1,3)

	def combine_heads(self, T):
		""" T: (batch, self.n_heads, n_nodes, self.head_depth)
			T transposed: (batch, n_nodes, self.n_heads, self.head_depth)
			return: (batch, n_nodes, self.embed_dim)
		"""
		T = T.permute(0,2,1,3).contiguous()
		shape = T.size()[:-2] + (self.embed_dim, )
		return T.view(*shape)

	def forward(self, x, mask = None):
		"""	q, k, v = x
			encoder arg x: [x, x, x]
			shape of q: (batch, n_nodes, embed_dim)
			output[0] - output[h_heads-1]: (batch, n_nodes, head_depth)
			--> concat output: (batch, n_nodes, head_depth * h_heads)
			return output: (batch, n_nodes, embed_dim)
		"""
		Q, K, V = x
		if self.need_W:
			Q, K, V = self.Wq(Q), self.Wk(K), self.Wv(V)
		Q, K, V = list(map(self.split_heads, [Q, K, V]))
		output = self.attention([Q, K, V], mask = mask)
		output = self.combine_heads(output)
		if self.need_W:
			return self.Wout(output)
		return output

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse and adj.is_sparse and adj.ndimension() == 2:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)