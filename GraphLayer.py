# @Time : 2021/4/9 11:09
# @Author : Scewiner, Xu, Mingzhou
# @File: Grapg_layer.py.py
# @Software: PyCharm
# from dgl.nn.pytorch import HeteroGraphConv
import torch
import torch.nn as nn
import fairseq.utils as utils
from fairseq.modules import LayerNorm
import torch.nn.functional as F
from .GraphDense import GCNLayer,GATLayer




class GraphLayer(nn.Module):
    def __init__(self, args):
        super(GraphLayer, self).__init__()
        self.drop = getattr(args, 'gcn_drop', 0.2)
        self.hidden_size = args.encoder_embed_dim
        self.num_head = args.encoder_attention_heads
        self.num_layers = args.ctx_layer
        self.activation_fn = utils.get_activation_fn(activation='tanh')
        self.activation_relu = utils.get_activation_fn(activation='relu')
        self.layers = nn.ModuleList([])
        self.layers.extend([
            self.build_layer('GCN') for _ in range(self.num_layers)
        ])
        self.GAT = self.build_layer('GAT')

    def build_layer(self, layer_type='GCN'):
        if layer_type == 'GCN':
            m = GCNLayer(self.hidden_size, self.hidden_size)
        elif layer_type == "GAT":
            m = GATLayer(self.hidden_size, self.hidden_size)
        else:
            raise NotImplementedError
        return m


    def forward(self, g_reps, graph, edge_type='forward'):
        residual = g_reps
        g = graph[0]
        if edge_type == 'backward':
            g = g.transpose(0,1)
#         g = g.to_dense()

        for layer in self.layers:
            g_reps = layer(g_reps,g)
            g_reps = self.activation_fn(g_reps)
        g_reps = F.dropout(g_reps, p=self.drop)
        g_reps = g_reps + residual
        g = graph[-1]
#         g = graph[1].to_dense()
        #         g = g.allow_zero_in_degree=True
        # making sentence representation
        output = self.GAT(g_reps,g)
        output = self.activation_relu(output)
        return output



class GraphEncoders(nn.Module):
    def __init__(self, args):
        super(GraphEncoders, self).__init__()
        self.shared_layers = getattr(args, "shared_layers", False)
        if self.shared_layers:
            self.layers = GraphLayer(args)
        else:
            self.layers = nn.ModuleList([])
            self.layers.extend([
                GraphLayer(args) for _ in range(2)
            ])
        self.ln_layers = nn.ModuleList([])
        self.ln_layers.extend([LayerNorm(args.encoder_embed_dim) for _ in range(2)])
        self.final_layer_ln = LayerNorm(args.encoder_embed_dim)
        self.edge_type = ['forward', 'backward']
        self.gate_units = GateUnit(args.encoder_embed_dim)
        self.drop = getattr(args, 'gcn_drop', 0.2)

    def forward(self, g_reps, graph):
        residual = g_reps
        output = []
        if self.shared_layers:
            for i in range(2):
                if i == 0:
                    x = self.ln_layers[i](g_reps)
                else:
                    x = self.ln_layers[i](residual)
                x = F.dropout(x, p=self.drop, training=self.training)
                edge_type = self.edge_type[i]
                x = self.layers(x, graph, edge_type)
                x = F.dropout(x, p=self.drop, training=self.training)
                output.append(x)
        else:
            for edge_type, ln_layer, layer in zip(self.edge_type, self.ln_layers, self.layers):
                x = ln_layer(g_reps) if edge_type == 'forward' else ln_layer(residual)
                x = F.dropout(x, self.drop, training=self.training)
                x = layer(x, graph, edge_type)
                x = F.dropout(x, p=0.1, training=self.training)
                output.append(x)
        output = self.gate_units(*output)
        output = self.final_layer_ln(output)
        return output



def build_linear(in_feat, out_feat, bias=True):
    m = nn.Linear(in_feat, out_feat, bias=bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0)
    return m


class GateUnit(nn.Module):
    def __init__(self, hidden_size):
        super(GateUnit, self).__init__()
        self.x_weight = build_linear(hidden_size, hidden_size)
        self.y_weight = build_linear(hidden_size, hidden_size)

    def forward(self, x, y):
        gate = torch.sigmoid(self.x_weight(x) + self.y_weight(y))
        return gate * x + (1 - gate) * y
