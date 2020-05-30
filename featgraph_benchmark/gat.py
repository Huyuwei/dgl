import torch
import torch.nn as nn
from gatconv import GATConv

import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden,
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(
                num_hidden, num_hidden,
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden, num_classes,
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h)
        return logits


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.sum().item(),
              val_mask.sum().item(),
              test_mask.sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    g = data.graph
    # add self loop
    # g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    # g.add_edges(g.nodes(), g.nodes())
    n_edges = g.number_of_edges()
    # create model
    model = GAT(g,
                args.n_layers,
                in_feats,
                args.n_hidden,
                n_classes,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    print(model)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # train
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            acc = evaluate(model, features, labels, test_mask)
            print("Epoch {:05d} | Time(s) {:.4f} | Test Accuracy {:.4f}". format(epoch, time.time() - t0, acc))
        else:
            print("Epoch {:05d} | Time(s) {:.4f}". format(epoch, time.time() - t0))

    # test
    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Final Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--n-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    print(args)

    main(args)
