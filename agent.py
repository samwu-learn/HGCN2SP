import torch
import torch.nn as nn
from models import Encoder, DecoderCell, CriticCell

class Agent(nn.Module):
    def __init__(self, policy_param, train_param, device):
        super().__init__()
        self.features_extractor = Encoder(
            policy_param['var_dim'], 
            policy_param['con_dim'],
            policy_param['l_hid_dim'],
            policy_param['scenario_dim'],
            policy_param['h_hid_dim'],
            policy_param['h_out_dim']
        )
        self.actor = DecoderCell(
            policy_param['scenario_dim'],
            policy_param['n_heads'],
            policy_param['clip']
        )
        self.critic = CriticCell()
        self.device = device
        self.decode_type = train_param['decode_type']
        self.cluster_k = train_param['sel_num']

    def get_action_and_value(self, x, action=None, decode_type="sampling", cluster_k=None, mode = "train"):
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
        if mode == "train":
            action, logprob, entropy = self.actor(
                self.device, encoder_output,
                return_cost=False,
                cluster_k=self.cluster_k,
                decode_type=self.decode_type,
                action=action
            )
            value = self.critic(batch_edge, batch_feat)
            return action, logprob, entropy, value
        elif mode == "test":
            action, logprob = self.actor(
                self.device, encoder_output,
                return_cost=True,
                cluster_k=self.cluster_k,
                decode_type=self.decode_type,
                action=action
            )            
            return action, logprob
        else:
            raise ValueError("mode should be 'train' or 'test'")