import torch
import torch.nn as nn
import fairseq.utils as utils
from fairseq.modules import LayerNorm
import torch.nn.functional as F
from .GraphDense import GCNLayer, GATLayer


class TypeGraphLayer(nn.Module):
    def __init__(self, args):
        super(TypeGraphLayer, self).__init__()
        self.edge_type = ['forward', 'backward', 'selfloop']
        self.drop = getattr(args, 'gcn_drop', 0.2)
        self.hidden_size = args.encoder_embed_dim

        self.activation_fn = utils.get_activation_fn(activation='tanh')

        self.layers = nn.ModuleList([])
        self.layers.extend([
            GCNLayer(self.hidden_size, self.hidden_size) for _ in range(len(self.edge_type))
        ])

        self.typeatt = getattr(args, 'typeatt', False)
        if self.typeatt:
            self.gate = TypeAttn(self.hidden_size)
        else:
            self.gate = GateUnit(self.hidden_size)
        self.final_layer_ln = LayerNorm(args.encoder_embed_dim)

    def forward(self, g_rep, graph):
        residual = g_rep  # GL x D
        outputs = []
        for edge_type, layer in zip(self.edge_type, self.layers):
            if edge_type is 'selfloop':
                continue
            if edge_type is 'backward':
                graph = graph.transpose(-2, -1)
            # print('{}:{}:{}'.format(edge_type,g_rep.size(),graph.size()))
            g_rep = layer(residual, graph)
            g_rep = self.activation_fn(g_rep)
            g_rep = F.dropout(g_rep, p=self.drop)
            outputs.append(g_rep.unsqueeze(1))  # GL x 1 x D in list
        outputs.append(residual.unsqueeze(1))  # residual as selfloop
        if self.typeatt:
            outputs = torch.cat(outputs, dim=1)

            final = self.gate(residual.unsqueeze(1), outputs, outputs)
        else:
            final = self.gate(*outputs)
        # print('final:{}'.format(final.size()))
        output = self.final_layer_ln(final)

        return output


class GraphEncoderT(nn.Module):
    def __init__(self, args):
        super(GraphEncoderT, self).__init__()
        self.share_ctx_layer = getattr(args, "share_ctx_layer", False)
        self.num_layers = args.ctx_layer
        if self.share_ctx_layer:
            self.layers = TypeGraphLayer(args)
        else:
            self.layers = nn.ModuleList([])
            self.layers.extend(
                [TypeGraphLayer(args) for _ in range(self.num_layers)]
            )
        self.ln_layers = nn.ModuleList([])
        self.ln_layers.extend(
            [LayerNorm(args.encoder_embed_dim) for _ in range(self.num_layers)]
        )
        self.gat_layer = GATLayer(args.encoder_embed_dim, args.encoder_embed_dim)
        self.activation_relu = utils.get_activation_fn(activation='relu')

    def forward(self, g_reps, graph):
        residual = g_reps
        g = graph[0]  # word-word graph
        if self.share_ctx_layer:
            for i in range(self.num_layers):

                g_reps = self.ln_layers[i](g_reps)
                # print('input:{}'.format(g_reps.size()))
                g_reps = self.layers(g_reps, g)
                # print('output:{}'.format(g_reps.size()))
                g_reps = F.dropout(g_reps, p=0.1, training=self.training)
                g_reps = g_reps + residual
                # print('LN:{}'.format(g_reps.size()))

        else:
            for ln_layer, layer in zip(self.ln_layers, self.layers):
                g_reps = ln_layer(g_reps)
                g_reps = layer(g_reps, g)
                g_reps = F.dropout(g_reps, p=0.1, training=self.training)
                g_reps = g_reps + residual

        # get sent nodes
        g = graph[-1]  # sent-word graph
        final = self.gat_layer(g_reps, g)
        final = self.activation_relu(final) # sent-level representation

        return final



class TypeAttn(nn.Module):
    def __init__(self, hidden_size):
        super(TypeAttn, self).__init__()
        self.q_w = build_linear(hidden_size, hidden_size)
        self.k_w = build_linear(hidden_size, hidden_size)
        self.v_w = build_linear(hidden_size, hidden_size)
        self.scaling = hidden_size ** -0.5

    def forward(self, Q, K, V):
        q = self.q_w(Q)  # Seq x 1 x D
        k = self.k_w(K)  # Seq x 3 x D
        v = self.v_w(V)  # Seq x 3 x D
        dim_size = Q.dim()
        if dim_size ==4:
            bzs,t,seq,dim = q.size()
            q = q.transpose(1,2).reshape(bzs*seq,-1,dim)
            k = k.transpose(1, 2).reshape(bzs*seq,-1,dim)
            v = v.transpose(1, 2).reshape(bzs*seq,-1,dim)
        attw = torch.bmm(q, k.transpose(1, 2))  # Seq x 1 x 3
        attw_float = utils.softmax( attw, dim=-1)
        attw = attw_float.type_as(attw)
        final = torch.bmm(attw, v)  # Seq x 1 x D
        if dim_size == 4:
            return final.reshape(bzs,seq,dim)
        return final.squeeze(1)


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
        self.z_weight = build_linear(hidden_size, hidden_size)

    def forward(self, x, y, z):
        # gate = torch.sigmoid(self.x_weight(x) + self.y_weight(y))
        g1 = torch.sigmoid(self.x_weight(x.squeeze(1)))
        g2 = torch.sigmoid(self.x_weight(y.squeeze(1)))
        g3 = torch.sigmoid(self.x_weight(z.squeeze(1)))
        return g1 * x.squeeze(1) + g2 * y.squeeze(1) + g3 * z.squeeze(1)
