import torch
import numpy as np
from torch import nn
from typing import Dict
import torch
from . import utils
import pandas as pd

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

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics)

    def to_csv(self, path: str) -> None:
        self.to_df().to_csv(path)

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

def power_iteration_find_hessian_eigen(loss, params, n_iter=30, tol=1e-4):
    """estimates the largest singular value based on power iteration"""
    # get number of params
    params = list(params)
    num_param = sum(p.numel() for p in params)
    # Calculate the gradient of the loss with respect to the model parameters
    #print(params)
    grad_params = torch.autograd.grad(loss, list(params), create_graph=True)
    #print("grad_params unfalttened:",grad_params)
    grad_params = torch.cat([e.flatten() for e in grad_params]) # flatten
    #print("grad_params:",grad_params)
    # Compute the vector product of the Hessian and a random vector using the power iteration method
    v = torch.rand(num_param).to(grad_params.device)
    v = v/torch.norm(v)
    #print(v)      
    Hv = torch.autograd.grad(grad_params, list(params), v, retain_graph=True)
    #print("Hv:",Hv)
    Hv = torch.cat([e.flatten() for e in Hv]) # flatten
    #print("Hv:",Hv)       
    # normalize Hv
    Hv = Hv /torch.norm(Hv)
    for i in range(n_iter):
        # Compute the vector product of the (inverse Hessian or) Hessian and Hv 
        w = torch.autograd.grad(grad_params, list(params), Hv, retain_graph=True)
        w = torch.cat([e.flatten() for e in w]) # flatten
        # Calculate the Rayleigh quotient to estimate the largest eigenvalue of the Hessian (inverse Hessian)
        eigenvalue = torch.dot(Hv, w)/ torch.dot(Hv, Hv) 
        # Check if the difference between consecutive estimates is below the tolerance level
        if i > 0 and torch.abs(eigenvalue - last_eigenvalue) < tol:
            break
        last_eigenvalue = eigenvalue
        # Update Hv for the next iteration
        Hv = w/torch.norm(w)        
    return eigenvalue.detach().item()

# def _compute_grad_vector(loss, params):
#     grad = torch.autograd.grad(loss, params, create_graph=False)
#     grad_vector = torch.cat([g.flatten() for g in grad])
#     return grad_vector

# def _hessian_vector_product(loss, params, vector):
#     grad = _compute_grad_vector(loss, params)
#     grad_vector_product = torch.autograd.grad(grad, params, vector, retain_graph=True)
#     grad_vector_product = torch.cat([g.flatten() for g in grad_vector_product])
#     return grad_vector_product

# def power_iteration_find_hessian_eigen(loss, params, v=None, n_iter=100, tol=1e-2):
#     # Ensure all params require gradients
#     params = list(params)
#     for p in params:
#         if not p.requires_grad:
#             raise ValueError("All parameters must have requires_grad=True")

#     # Ensure loss requires gradient
#     if not loss.requires_grad:
#         raise ValueError("Loss must have requires_grad=True")

#     # Check the gradient function of the loss
#     if loss.grad_fn is None:
#         raise ValueError("Loss must be part of a computation graph and have a grad_fn")

#     if v is None:
#         v = torch.randn_like(torch.cat([p.flatten() for p in params]))
#     v = v / torch.norm(v)
#     for _ in range(n_iter):
#         Hv = _hessian_vector_product(loss, params, v)
#         Hv_norm = torch.norm(Hv)
#         v = Hv / Hv_norm
#     return v, Hv_norm.item()

def grad_error_metrics(model: utils.QuantWrapper, quant_scheme, data, target, iters=100):
    # returns : < mini-batch-grad , E[error] > , E[||error||^2]

    model.apply_quant_scheme(utils.FP32_SCHEME)
    loss = model.module.loss_acc(data, target)["loss"]
    full_grads = torch.autograd.grad(loss, model.parameters())
    full_grad_vector = torch.cat([g.flatten() for g in full_grads]).detach()

    model.apply_quant_scheme(quant_scheme)
    grads_acc = torch.zeros_like(full_grad_vector)
    error_norm_acc = 0

    for _ in range(iters):
        loss = model.module.loss_acc(data, target)["loss"]
        grad = torch.autograd.grad(loss, model.parameters())
        grad_vector = torch.cat([g.flatten() for g in grad]).detach()
        error_norm_acc += (grad_vector - full_grad_vector).norm().item()
        grads_acc += grad_vector
    grad_mean = grads_acc / iters
    exp_error_norm = error_norm_acc / iters
    grad_bias = grad_mean - full_grad_vector
    return torch.dot(full_grad_vector, grad_bias).item(), exp_error_norm

def grad_bias_deterministic_deterministic(model, scheme, data, target):
    # returns : < mini-batch-grad , E[error] > , E[||error||^2]
    model.apply_quant_scheme(scheme)
    loss = model.module.loss_acc(data, target)["loss"]
    grads = torch.autograd.grad(loss, model.parameters())
    grad_vec = torch.cat([g.flatten() for g in grads])
    model.apply_quant_scheme(utils.FP32_SCHEME)
    loss = model.module.loss_acc(data, target)["loss"]
    grads = torch.autograd.grad(loss, model.parameters())
    full_grad_vec = torch.cat([g.flatten() for g in grads])
    return torch.dot(full_grad_vec, grad_vec - full_grad_vec).item(), torch.norm(grad_vec - full_grad_vec).item()





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
