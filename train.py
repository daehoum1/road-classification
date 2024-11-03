import torch.nn.functional as F
import torch

def train(model, optimizer, dataset_ind, criterion, device):
    model.train()
    optimizer.zero_grad()
    loss = model.loss_compute(dataset_ind, criterion, device)
    loss.backward()
    optimizer.step()
    return loss

def train_batch(model, optimizer, dataset_ind, device, train_loader):
    model.train()
    total_loss = 0
    train_idx = dataset_ind.splits['train']
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(dataset_ind.x[n_id], adjs, True)
        mask = torch.isin(n_id[:batch_size].to(device), train_idx)
        loss = F.cross_entropy(out[:batch_size][mask], dataset_ind.y[n_id[:batch_size]].squeeze(1)[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)