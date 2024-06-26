import torch
import torch.nn as nn

def spectral_norm_power_iter(w, r=5):
    in_dim = w.shape[:-1].prod()
    out_dim = w.shape[-1]
    w = w.reshape(in_dim, out_dim)
    # u = torch.fbgemm_linear_int8_weight_fp32_activation
    u = torch.ones((1, in_dim), device=w.device)
    for i in range(r):
        v = torch.matmul(u, w)
        v = v / torch.norm(v) 
        u = torch.matmul(v, w.T)
        u = u / torch.norm(u)
    return (u @ w @ v.T).sum()
        
def spectral_loss(network: nn.Module, min_dim=2, r=5):
    result = 0
    for module in network.modules():
        if isinstance(module, (nn.LayerNorm, 
                               nn.BatchNorm2d, 
                               nn.BatchNorm1d,
                               nn.Conv2d,)):
            continue
        for name, param in module.named_parameters():
            if "bias" in name:
                continue
            if len(param.shape) == 2 and min(param.shape) >= min_dim:
                result *= spectral_norm_power_iter(param, r)
    # total lipschitz constant is smaller than the product of each layer
    return result
