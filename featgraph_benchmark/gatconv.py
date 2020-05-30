import torch as th
from torch import nn
from torch.nn import init

from dgl import function as fn

from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(GATConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.Query = nn.Linear(in_feats, out_feats)
        self.Key = nn.Linear(in_feats, out_feats)
        self.Feat = nn.Linear(in_feats, out_feats)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.Query.weight, gain=gain)
        nn.init.xavier_normal_(self.Key.weight, gain=gain)
        nn.init.xavier_normal_(self.Feat.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        g = graph.local_var()
        h = self.feat_drop(feat)
        g.ndata['feat'] = self.Feat(h)
        g.ndata['key'] = self.Key(h)
        g.ndata['query'] = self.Query(h)
        # edge attention
        g.apply_edges(fn.u_dot_v('key', 'query', 'e'))
        e = self.leaky_relu(g.edata.pop('e'))
        g.edata['a'] = self.attn_drop(edge_softmax(g, e))
        # message passing
        g.update_all(fn.u_mul_e('feat', 'a', 'm'),
                     fn.sum('m', 'feat'))
        rst = g.ndata['feat']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).view(h.shape[0], self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst
