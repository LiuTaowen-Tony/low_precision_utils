import torch
import numpy as np
from torch import nn
from typing import Dict
import torch

class EMAMetrics:
    def __init__(self, beta=0.8) -> None:
        self.metrics = {}
        self.beta = beta

    def update(self, metrics: Dict[str, float]) -> Dict[str, float]:
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if key not in self.metrics:
                self.metrics[key] = value
            else:
                self.metrics[key] = self.beta * self.metrics[key] + (1 - self.beta) * value

        report = {f"{key}_ema": value for key, value in self.metrics.items()}
        report["ema_beta"] = self.beta
        return report

class Logger:
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.metrics = {}
        self.n_iter = 0

    def log(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = [value]
            else:
                self.metrics[key].append(value)
        self.n_iter += 1

    def report(self) -> Dict[str, float]:
        report = {}
        for key, values in self.metrics.items():
            report[key] = np.mean(values)
        return report

    def should_log(self) -> bool:
        return self.n_iter % self.log_interval == 0

    def reset(self) -> None:
        self.metrics = {}
        self.n_iter = 0
    def __getitem__(self, key):
        return self.metrics[key]
    def __setitem__(self, key, value):
        self.metrics[key] = value
    def __contains__(self, key):
        return key in self.metrics
    def __len__(self):
        return len(self.metrics)



def diff_of_grad(wrapper, model_weight, master_weight, data, target):
    # at the same traning step
    # that is have the same training data and target
    # we compute the gradient on full precison model
    # then compute the same thing on low precision model (with different seed)
    # we check if for each parameter the estimation is biased

    # this time, we check the difference between activation quantise only
    # and full precision model

    master_weight.zero_grad()
    reference_loss = master_weight.loss_acc(data, target)["loss"]
    reference_loss.backward()
    reference_grad = get_grad(master_weight)
    master_weight.zero_grad()

    grad_estimation_samples = []
    for i in range(100):
        model_weight.zero_grad()
        sample_loss = model_weight.loss_acc(data, target)["loss"]
        sample_loss.backward()
        sample_grad = get_grad(model_weight)
        grad_estimation_samples.append(sample_grad)
        model_weight.zero_grad()

    return reference_grad, sample_grad
    



def compute_grad_weight_corr(model):
    result = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # corrcoef takes 1 input
            # we need to flatten the tensor
            # and stack
            corr_input = torch.stack([param.grad.data.flatten().abs(), param.data.flatten().abs()])
            corr = torch.corrcoef(corr_input)
            # corr = torch.corrcoef(param.grad.data.flatten(), param.data.flatten())[0, 1]
            result[f"{name}_grad_corr"] = corr[0, 1].item()
    return result
    


    

def grad_on_dataset(network, data, target):
    network.train()
    total_norm = 0
    network.zero_grad()
    idx = torch.randperm(len(data))
    data = data[idx]
    target = target[idx]
    loss_acc = network.loss_acc(data, target)
    loss = loss_acc["loss"]
    acc = loss_acc["acc"]
    loss.backward()
    total_norm = nn.utils.clip_grad_norm_(network.parameters(), float('inf'))
    grad_zero = grad_zero_percentage(network)
    # grad_weight_corr = compute_grad_weight_corr(network)
    network.zero_grad()
    return {"grad_norm_entire": total_norm.item()} #| grad_zero #| grad_weight_corr


def grad_zero_percentage(network):
    n_params = 0
    for param in network.parameters():
        n_params += param.numel()
        n_zero = torch.sum(param.grad.data == 0).item()
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
