from abc import abstractmethod
import json
import torch
from torch import nn

import argparse
import qtorch
import qtorch.quant
import copy
from . import layers

import torch.autograd
import torch.nn.grad
from dataclasses import dataclass

SCALING=True
TEST_NAN = False

def assert_nan(x):
    if not TEST_NAN:
        return x
    if x.isnan().any():
        print(x)
    assert torch.all(x.isnan() == False)
    assert torch.all(x.isinf() == False)
    return x
import torch

def add_argparse(parser : argparse.ArgumentParser):
    parser.add_argument("--quant_scheme", type=json.loads, default="{}")
    return parser

class QuantMethod:
    @abstractmethod
    def quant(self, x):
        pass

    @classmethod
    def from_dict(cls, json_dict: dict):
        return cls(**json_dict)

    @classmethod
    def from_json(cls, json_dict: dict):
        number_type = json_dict.get("number_type", "fp")
        json_dict["number_type"] = number_type
        json_dict = json_dict.copy()
        del json_dict["number_type"]

        if number_type == "fp":
            return FPQuant.from_dict(json_dict)
        elif number_type == "int":
            return IntQuant.from_dict(json_dict)
        elif number_type == "block":
            return BlockQuant.from_dict(json_dict)
        elif number_type == "scaled_int":
            return ScaledIntQuant.from_dict(json_dict)
        else:
            raise ValueError(f"number_type {number_type} not recognized")

# quant method strategy
@dataclass
class FPQuant(QuantMethod):
    exp : int = 8
    man : int = 23
    round_mode : str = "stochastic"

    def quant(self, x):
        if self.exp == 8 and self.man == 23:
            return x
        return qtorch.quant.float_quantize(x, self.exp, self.man, self.round_mode)

@dataclass
class IntQuant(QuantMethod):
    wl : int = 8
    fl : int = 16
    clamp : bool = True
    symmetric : bool = False
    round_mode : str = "stochastic"

    def quant(self, x):
        return qtorch.quant.fixed_point_quantize(x, self.wl, self.fl, self.clamp, self.symmetric, self.round_mode)

@dataclass
class BlockQuant(QuantMethod):
    wl : int = 8
    dim : int = 8
    round_mode : str = "stochastic"

    def quant(self, x):
        return qtorch.quant.block_quantize(x, self.wl, self.dim, self.round_mode)

@dataclass
class ScaledIntQuant(QuantMethod):
    wl : int = 8
    clamp : bool = True
    symmetric : bool = False
    round_mode : str = "stochastic"

    def quant(self, x):
        x_scale = x.abs().max()
        x = x / x_scale
        result =  qtorch.quant.fixed_point_quantize(x, self.wl, self.wl - 1, self.clamp, self.symmetric, self.round_mode)
        return result * x_scale

class QuantScheme:
    def __init__(self, 
                act: QuantMethod,
                weight: QuantMethod,
                bact: QuantMethod,
                bweight: QuantMethod,
                goact: QuantMethod,
                goweight: QuantMethod,
                same_input: bool = False,
                same_weight: bool = False):
        self.act = act
        self.goact = goact
        self.weight = weight
        self.bact = bact
        self.bweight = bweight
        self.goweight = goweight
        self.same_input = same_input
        self.same_weight = same_weight

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        json_dict = args.quant_scheme
        fp_default = {"number_type":"fp"}
        return QuantScheme(
            act=QuantMethod.from_json( json_dict.get("act", fp_default)),
            weight=QuantMethod.from_json( json_dict.get("weight", fp_default)),
            bact=QuantMethod.from_json( json_dict.get("bact", fp_default)),
            bweight=QuantMethod.from_json( json_dict.get("bweight", fp_default)),
            goact=QuantMethod.from_json( json_dict.get("goact", fp_default)),
            goweight=QuantMethod.from_json( json_dict.get("goweight", fp_default)),
            same_input=json_dict.get("same_input", False),
            same_weight=json_dict.get("same_weight", False)
        )

    def __str__(self):
        return self.__dict__.__str__()

FP32 = FPQuant()
FP32_SCHEME = QuantScheme(
    FP32, FP32, FP32, FP32, FP32, FP32,
    same_input=True,
    same_weight=True
)

class QuantWrapper(nn.Module):
    def __init__(self, module, quant_scheme):
        super(QuantWrapper, self).__init__()
        module = replace_with_quantized(module, quant_scheme)
        self.quant_scheme = quant_scheme
        self.module = module

    def apply_quant_scheme(self, quant_scheme, filter=None):
        apply_quant_scheme(self, quant_scheme, filter)

    def forward(self, *args, **kw):
        return self.module(*args, **kw)

    def loss_acc(self, X, y):
        return self.module.loss_acc(X, y)

def apply_quant_scheme(network, quant_scheme, filter=None):
    for name, module in network.named_modules():
        if hasattr(module, "quant_scheme"):
            if filter:
                if filter(name, module):
                    module.quant_scheme = quant_scheme
    return network

def replace_with_quantized(network, quant_scheme):
    to_replace = []
    for name, module in network.named_children():
        if isinstance(module, nn.Linear):
            new_module = layers.QuantLinear.from_full_precision(module, quant_scheme)
            to_replace.append((name, new_module))
        elif isinstance(module, nn.Conv2d):
            new_module = layers.QuantConv2d.from_full_precision(module, quant_scheme)
            to_replace.append((name, new_module))
        elif isinstance(module, nn.Conv1d):
            new_module = layers.QuantConv1d.from_full_precision(module, quant_scheme)
            to_replace.append((name, new_module))
        else:
            replace_with_quantized(module, quant_scheme)
    
    for name, new_module in to_replace:
        setattr(network, name, new_module)

    return network


class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
