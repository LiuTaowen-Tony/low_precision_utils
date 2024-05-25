import torch
from torch import nn


def grad_on_batch(network, data, target, loss):
    network.train()
    total_norm = 0
    network.zero_grad()
    output = network(data)
    loss = loss(output, target)
    loss.backward()
    total_norm = nn.utils.clip_grad_norm_(network.parameters(), float('inf'))
    network.zero_grad()
    network.eval()
    return {"grad_norm_batch": total_norm.item()}

def grad_on_trainset(network, dataset, batch_size, criterion):
    network.train()
    total_norm = 0
    network.zero_grad()
    idx = torch.randperm(len(data))
    idx = idx[:max(len(idx), 2048)]
    for n, i in enumerate(range(0, len(idx), batch_size)):
        data, target = dataset[idx[i:i + batch_size]]
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
    total_norm = nn.utils.clip_grad_norm_(network.parameters(), float('inf'))
    total_norm /= n
    network.zero_grad()
    return {"grad_norm_entire": total_norm.item()}