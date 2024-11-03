from backbone import *
from torch_geometric.nn.models import GIN
import torch
import torch.nn.functional as F
from torch import nn

class Model(nn.Module):
    def __init__(self, d, c, args):
        super(Model, self).__init__()
        if args.method == 'gcn':
            self.encoder = GCN(in_channels=d,
                               hidden_channels=args.hidden_channels,
                               out_channels=c,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=args.use_bn)
        elif args.method == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                               out_channels=c, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.method == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout, use_bn=args.use_bn)
        elif args.method == 'mixhop':
            self.encoder = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.method == 'gcnjk':
            self.encoder = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.method == 'gatjk':
            self.encoder = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.method == 'graphsage_mean':
            self.encoder = GraphSAGENet_mean(d, args.hidden_channels, c)
        elif args.method == 'graphsage_meanpool':
            self.encoder = GraphSAGENet_meanpool(d, args.hidden_channels, c)
        elif args.method == 'graphsage_maxpool':
            self.encoder = GraphSAGENet_maxpool(d, args.hidden_channels, c)
        elif args.method == 'graphsage_lstm':
            self.encoder = GraphSAGENet_lstm(d, args.hidden_channels, c)
        elif args.method == 'gin':
            self.encoder = GIN(d, args.hidden_channels, out_channels=c, num_layers = args.num_layers, dropout=args.dropout, train_eps = True)
        elif args.method == 'gain':
            self.encoder = GAIN(d, args.hidden_channels, out_channels=c, num_layers = args.num_layers, dropout=args.dropout)
        elif args.method == 'rfn':
            self.encoder = RFN(d, args.hidden_channels, out_channels=c, num_layers = args.num_layers, dropout=args.dropout)
        elif args.method == 'DAE':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                               out_channels=c, num_layers=args.num_layers,
                               dropout=args.dropout)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device, flag=False):
        if flag == False:
            x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        else:
            x, edge_index = dataset, torch.cat([device[0].edge_index, device[1].edge_index],dim=-1)
        return self.encoder(x, edge_index)

    def loss_compute(self, dataset_ind, criterion, device):
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        logits_in = self.encoder(x_in, edge_index_in)
        train_in_idx = dataset_ind.splits['train']
        pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
        sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))
        loss = sup_loss
        return loss