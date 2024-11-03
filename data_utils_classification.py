import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add
import torch.nn as nn

def rand_splits(node_idx, train_prop=.5, valid_prop=.25):
    """ randomly splits label into train/valid/test splits """
    splits = {}
    n = node_idx.size(0)

    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    splits['train'] = node_idx[train_indices]
    splits['valid'] = node_idx[val_indices]
    splits['test'] = node_idx[test_indices]

    return splits

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_pred.shape == y_true.shape:
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

@torch.no_grad()
def evaluate_classify(model, dataset, eval_func, criterion, args, device):
    model.eval()

    train_idx, valid_idx, test_idx = dataset.splits['train'], dataset.splits['valid'], dataset.splits['test']
    y = dataset.y
    out = model(dataset, device).cpu()

    if args.use_prop == False:
        train_score = eval_func(y[train_idx], out[train_idx])
        valid_score = eval_func(y[valid_idx], out[valid_idx])
        test_score = eval_func(y[test_idx], out[test_idx])
    else:
        out = nn.functional.softmax(out, dim=-1).to(device)
        out = propagation_att(out, dataset.edge_index.to(device), prop_layers=args.K, alpha=args.alpha).cpu()
        train_score = eval_func(y[train_idx], out[train_idx])
        valid_score = eval_func(y[valid_idx], out[valid_idx])
        test_score = eval_func(y[test_idx], out[test_idx])
    if args.use_prop == False:
        valid_out = F.log_softmax(out[valid_idx], dim=1)
        valid_loss = criterion(valid_out, y[valid_idx].squeeze(1).cpu())
    else:
        valid_loss = criterion(out[valid_idx], y[valid_idx].squeeze(1).cpu())
    return train_score, valid_score, test_score, valid_loss

def evaluate_classify_batch(model, dataset, eval_func, criterion, args, device, test_loader):
    model.eval()
    train_idx, valid_idx, test_idx = dataset.splits['train'], dataset.splits['valid'], dataset.splits['test']
    y = dataset.y
    # out = model(dataset, device).cpu()
    out_list = []
    for batch_size, n_id, adjs in test_loader:
        adjs = [adj.to(device) for adj in adjs]
        out = model(dataset.x[n_id], adjs, True)
    if args.use_prop == False:
        train_score = eval_func(y[train_idx], out[train_idx])
        valid_score = eval_func(y[valid_idx], out[valid_idx])
        test_score = eval_func(y[test_idx], out[test_idx])
    else:
        out = nn.functional.softmax(out, dim=-1).to(device)
        out = propagation_att(out, dataset.edge_index.to(device), prop_layers=args.K, alpha=args.alpha, beta = args.beta).cpu()
        train_score = eval_func(y[train_idx], out[train_idx])
        valid_score = eval_func(y[valid_idx], out[valid_idx])
        test_score = eval_func(y[test_idx], out[test_idx])
    if args.use_prop == False:
        valid_out = F.log_softmax(out[valid_idx], dim=1).cpu()
        valid_loss = criterion(valid_out, y[valid_idx].squeeze(1).cpu())
    else:
        valid_loss = criterion(out[valid_idx], y[valid_idx].squeeze(1).cpu())

    return train_score, valid_score, test_score, valid_loss

def convert_to_adj(edge_index,n_node):
    '''convert from pyg format edge_index to n by n adj matrix'''
    adj=torch.zeros((n_node,n_node))
    row,col=edge_index
    adj[row,col]=1
    return adj

def propagation_att(e_0, edge_index, prop_layers=1, alpha=0.5):
    N = e_0.shape[0]
    row, col = edge_index
    for _ in range(prop_layers):
        edge_weight = torch.sum((e_0[row] * e_0[col]), dim=-1)
        deg_W = scatter_add(edge_weight, row, dim_size = N)
        deg_W_inv = deg_W.pow_(-1.0)
        deg_W_inv.masked_fill_(deg_W_inv == float("inf"), 0)
        A_Dinv = edge_weight * deg_W_inv[row]
        adj = torch.sparse.FloatTensor(edge_index, values = A_Dinv, size=[N,N]).to(edge_index.device)
        e_0 = torch.sparse.mm(adj, e_0) * (1 - alpha) + alpha * e_0
    return e_0.squeeze(1)