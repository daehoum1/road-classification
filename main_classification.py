import argparse
import random
from dataset import load_dataset
import copy
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from models import *
from train import train, train_batch
from torch_geometric.loader import NeighborSampler
from data_utils_classification import rand_splits, evaluate_classify,evaluate_classify_batch, eval_acc
import os
from sklearn.preprocessing import MinMaxScaler

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='road1', help='road1: linkoping / road2: suwon / road3: oakland')
    parser.add_argument('--test_dataset', type=str, default='road1')
    parser.add_argument('--transductive', type=bool, default=True)
    parser.add_argument('--runs', type=int, default=5, help='number of distinct runs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default="mlp")
    parser.add_argument('--use_prop', type=bool, default=True)
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--use_bn', type=bool, default=False)

parser = argparse.ArgumentParser()
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '0'

G, feats, id_map, walks, class_map = load_dataset(args.dataset)
if args.transductive == False:
    G_test, feats_test, id_map_test, walks_test, class_map_test = load_dataset(args.test_dataset)

device = torch.device("cuda:" + "0") if torch.cuda.is_available() else torch.device("cpu")

dataset_ind = from_networkx(G)
dataset_ind = Data(x=torch.from_numpy(feats).type(torch.float32), y=torch.tensor(list(class_map.values())).unsqueeze(dim=-1), edge_index=dataset_ind.edge_index.to(device)).to(device)
dataset_ind.node_idx = torch.nonzero(dataset_ind.y.squeeze()<6).squeeze()

if args.transductive == False:
    dataset_ind_test = from_networkx(G_test)
    dataset_ind_test = Data(x=torch.from_numpy(feats_test).type(torch.float32), y=torch.tensor(list(class_map_test.values())).unsqueeze(dim=-1), edge_index=dataset_ind_test.edge_index.to(device)).to(device)
    dataset_ind_test.node_idx = torch.nonzero(dataset_ind_test.y.squeeze()<6).squeeze()

c = int(max(dataset_ind.y+ 1))
d = dataset_ind.x.shape[1]
if args.method == 'DAE':
    d = dataset_ind.x.shape[1] + 8
else:
    model = Model(d, c, args).to(device)

criterion = nn.NLLLoss()
eval_func = eval_acc

if args.method != 'lp':
    model.train()
    print('MODEL:', model)

train_accs, val_accs, test_accs= [], [], []

if args.method == 'DAE':
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_ind.x = torch.from_numpy(scaler.fit_transform(dataset_ind.x.cpu().numpy())).to(device)

for run in range(args.runs):
    fix_seed(run)
    dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=0.8, valid_prop=0.1)
    if args.transductive == False:
        dataset_ind_test.splits = rand_splits(dataset_ind_test.node_idx, train_prop=0.8, valid_prop=0.1)

    if args.method == "lp":
        model = get_lp(
            args
        ).to(device)
        dataset_ind.train_mask = torch.zeros((dataset_ind.num_nodes), dtype=torch.bool)
        dataset_ind.train_mask[dataset_ind.splits['train']] = True
        out = model(y=dataset_ind.y, edge_index=dataset_ind.edge_index, mask=dataset_ind.train_mask)
        train_idx, valid_idx, test_idx = dataset_ind.splits['train'], dataset_ind.splits['valid'], dataset_ind.splits['test']
        y = dataset_ind.y
        train_score = eval_func(y[dataset_ind.splits['train']], out[dataset_ind.splits['train']])
        valid_score = eval_func(y[dataset_ind.splits['valid']], out[dataset_ind.splits['valid']])
        test_acc = eval_func(y[dataset_ind.splits['test']], out[dataset_ind.splits['test']])
    else:
        if args.method == 'DAE':
            assert args.transductive == True
            x_original = copy.copy(dataset_ind.x)
            extracted_x = extract_feat(dataset_ind.x)
            dataset_ind.x = torch.cat([x_original, extracted_x.detach()],dim=-1)
        model.reset_parameters()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = float('-inf')
        auroc=0
        aupr=0
        fpr95=1.0
        val_accs = []
        if args.method in ["gain", "graphsage_mean", "graphsage_meanpool", "graphsage_maxpool", "graphsage_lstm"]:
            train_loader = NeighborSampler(dataset_ind.edge_index, sizes = [9,3], batch_size = 1024, shuffle = True, num_nodes=dataset_ind.num_nodes)
            if args.transductive == True:
                test_loader = NeighborSampler(dataset_ind.edge_index, sizes = [9,3], batch_size = dataset_ind.num_nodes, shuffle = False, num_nodes=dataset_ind.num_nodes)
            else:
                test_loader = NeighborSampler(dataset_ind_test.edge_index, sizes = [9,3], batch_size = dataset_ind_test.num_nodes, shuffle = False, num_nodes=dataset_ind_test.num_nodes)
        for epoch in range(args.epochs):
            if args.method in ["gain", "graphsage_mean", "graphsage_meanpool", "graphsage_maxpool", "graphsage_lstm"]:
                loss = train_batch(model, optimizer, dataset_ind, device, train_loader)
            else:
                loss = train(model, optimizer, dataset_ind, criterion, device)
            if args.method in ["gain", "graphsage_mean", "graphsage_meanpool", "graphsage_maxpool", "graphsage_lstm"]:
                if args.transductive == True:
                    result = evaluate_classify_batch(model, dataset_ind, eval_func, criterion, args, device, test_loader)
                else:
                    result = evaluate_classify_batch(model, dataset_ind_test, eval_func, criterion, args, device, test_loader)
            else:
                if args.transductive == True:
                    result = evaluate_classify(model, dataset_ind, eval_func, criterion, args, device)
                else:
                    result = evaluate_classify(model, dataset_ind_test, eval_func, criterion, args, device)
            if epoch == 0 or result[1] > max(val_accs):
                test_acc = result[2]
            val_accs.append(result[1])
    test_accs.append(test_acc)
mean, std = np.mean(test_accs), np.std(test_accs)
print(f"ACC: {mean * 100:.2f}% +- {std * 100:.2f}")