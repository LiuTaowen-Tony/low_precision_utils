from typing import Dict
from torch import nn
import torch

class EMAMetrics:
    def __init__(self, beta=0.9) -> None:
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
    network.zero_grad()
    return {"grad_norm_entire": total_norm.item()}