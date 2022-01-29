# @Time : 2021/5/15 16:54
# @Author : Scewiner, Xu, Mingzhou
# @File: GraphDense.py.py
# @Software: PyCharm
import torch
import torch.nn as nn



class GCNLayer(nn.Module):
    def __init__(self,
                 in_feature: int,
                 out_feature: int,
                 bias: bool = True
                 ):
        super(GCNLayer, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.for_linear = Linear(self.in_feature, self.out_feature, bias)

    def forward(self, nodes: torch.Tensor, graph: torch.FloatTensor) -> torch.Tensor:
        '''
        '''
        x = self.for_linear(nodes)
        x = x.contiguous()
        # bzs = x.size(0)
        # dim = x.size(-1)

        assert graph.dim() == x.dim()

        deg = self.degree(graph)
        if graph.dim()==3:
            x = torch.bmm(deg,x)
        else:
            x = torch.mm(deg, x)

        return x

    def degree(self, graph: torch.Tensor) -> torch.Tensor:
        mask_self_loop = torch.eye(graph.size(-1)).bool().cuda()
        if graph.dim() == 3:
            mask_self_loop = mask_self_loop.unsqueeze(0)
        graph = graph.masked_fill(mask_self_loop, 0)
        deg = graph.sum(-1).clamp(min=1).pow(-0.5)

        return deg.unsqueeze(-1) * graph * deg.unsqueeze(-2)


class GATLayer(nn.Module):
    def __init__(self,
                 in_feature: int,
                 out_feature: int,
                 heads: int = 8,
                 bias: bool = True
                 ):
        super(GATLayer, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.heads = heads
        self.for_linear = Linear(self.in_feature, self.out_feature, bias)

    def forward(self, nodes: torch.Tensor, graph: torch.FloatTensor) -> torch.Tensor:
        '''
        '''
        x = self.for_linear(nodes)
        dim_size = x.dim()
        if dim_size ==3:
            l,seq,d = x.size()
            x = x.reshape(l, seq, self.heads, d // self.heads).contiguous().transpose(1,2)
            x = x.reshape(l*self.heads,seq,d//self.heads) # bzs*8,len,dim/8
        else:
            l, d = x.size()
            x = x.reshape(l, self.heads, d//self.heads)
            x = x.contiguous().transpose(0, 1) #8,sl,d
        # print(graph.size())
        deg = self.degree(graph) #b,tl,sl
        # print(deg.size())
        m,n = deg.size()
        deg = deg.unsqueeze(1).repeat(self.heads,1,1).reshape(-1,m,n)
        if dim_size == 3:
            deg = deg.reshape(-1,1,n)
        x = torch.bmm(deg, x)

        x = x.reshape(self.heads,-1,d//self.heads).contiguous().transpose(0,1).reshape(-1,d)
        return x

    def degree(self, graph: torch.Tensor) -> torch.Tensor:
        # graph = graph.masked_fill(torch.eye(graph.size(-1)).unsqueeze(0).bool().cuda(), 1)
        deg = graph.sum(-1).clamp(min=1).pow(-0.5)

        return deg.unsqueeze(-1) * graph * deg.unsqueeze(-1)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
