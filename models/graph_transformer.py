import torch
import torch.nn as nn

class Graph_Transformer(nn.Module):
    def __init__(self, encoder_model, decoder_model, device, cluster_k = 3, decode_type = 'sampling', return_pi = True):
        super(Graph_Transformer, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.device = device
        self.return_pi = return_pi
        self.k = cluster_k
        self.decode_type = decode_type

    def forward(self, seq1, seq2, number, adj, edge_adj, sparse, cls, decode_type = 'sampling'):
        self.decode_type = decode_type
        encoder_output = self.encoder(seq1, seq2, number, adj, edge_adj, sparse)
        cost, ll, pi = self.decoder(self.device, encoder_output, self.return_pi, self.k, self.decode_type, cls)
        return cost, ll, pi
