import torch
import numpy as np
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



def grad_zero_percentage(network):
    n_params = 0
    for param in network.parameters():
        n_params += param.numel()
        n_zero = torch.sum(param == 0).item()
    return {"zero_percentage": n_zero / n_params}


def grad_on_trainset(network, dataset, batch_size, criterion):
    network.train()
    total_norm = 0
    network.zero_grad()
    dataset = torch.utils.data.Subset(dataset, range(0, 2048))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    zero_percents = []
    for n, (data, target) in enumerate(dataloader):
        data = data.cuda()
        target = target.cuda()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        total_norm += nn.utils.clip_grad_norm_(network.parameters(), float('inf'))
        zero_percents.append(grad_zero_percentage(network)["zero_percentage"])
        network.zero_grad()
    total_norm /= n
    return {"grad_norm_entire": total_norm.item(), 
            "zero_percentage": np.mean(zero_percents)}