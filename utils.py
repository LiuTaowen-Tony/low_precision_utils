import json
import torch
from torch import nn
import torch.autograd
import torch.nn.grad

import argparse
import qtorch
import qtorch.quant
import copy

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
from torch.autograd import Function

class quant_linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, quant_scheme:"QuantScheme" = None):
        ctx.quant_scheme = quant_scheme
        input_shape = input.shape
        # input = input.view(input_shape[0], -1)
        input = input.view(-1, input_shape[-1])
        input_type = input.dtype
        qinput = quant_scheme.act.quant(input)
        qweight = quant_scheme.weight.quant(weight)
        if bias is not None:
            bias = bias.to(torch.bfloat16)

        if quant_scheme.same_input:
            input = qinput
        if quant_scheme.same_weight:
            weight = qweight
        ctx.save_for_backward(qinput, qweight, bias)

        output = qinput.mm(qweight.t()).to(input_type)
        if bias is not None:
            output += bias
        return output.view(*input_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        quant_scheme : QuantScheme = ctx.quant_scheme
        grad_output_shape = grad_output.shape
        # print(grad_output_shape)
        grad_output = grad_output.reshape(-1, grad_output_shape[-1])
        grad_output_type = grad_output.dtype
        qinput, qweight, bias = ctx.saved_tensors

        if not quant_scheme.same_input:
            qinput = quant_scheme.bact.quant(qinput)
        if not quant_scheme.same_weight:
            qweight = quant_scheme.bweight.quant(qweight)

        qgrad_output1 = quant_scheme.goact.quant(grad_output)
        qgrad_output2 = quant_scheme.goweight.quant(grad_output)

        grad_input = qgrad_output1.mm(qweight).to(grad_output_type)
        grad_weight = qgrad_output2.t().mm(qinput).to(grad_output_type)

        grad_bias = grad_output.sum(0) if bias is not None else None
        return grad_input.view(*grad_output_shape[:-1], -1), grad_weight, grad_bias, None

class quant_conv1d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, quant_scheme:"QuantScheme"):
        ctx.quant_scheme = quant_scheme
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        qinput = quant_scheme.act.quant(input)
        qweight = quant_scheme.weight.quant(weight)

        if quant_scheme.same_input:
            input = qinput
        if quant_scheme.same_weight:
            weight = qweight
        ctx.save_for_backward(qinput, qweight, bias)

        output = torch.nn.functional.conv1d(qinput, qweight, bias, stride, padding, dilation, groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        quant_scheme : QuantScheme = ctx.quant_scheme
        qinput, qweight, bias = ctx.saved_tensors

        if not quant_scheme.same_input:
            qinput = quant_scheme.bact.quant(qinput)
        if not quant_scheme.same_weight:
            qweight = quant_scheme.bweight.quant(qweight)

        qgrad_output = quant_scheme.goact.quant(grad_output)
        qgrad_output2 = quant_scheme.goweight.quant(grad_output)
        
        grad_input = torch.nn.grad.conv1d_input(
            qinput.shape, qweight, qgrad_output, 
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        grad_weight = torch.nn.grad.conv1d_weight(
            qinput, qweight.shape, qgrad_output2, 
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        grad_bias = grad_output.sum(dim=(0, 2)) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class quant_conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, quant_scheme:"QuantScheme"):
        ctx.quant_scheme = quant_scheme
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        qinput = quant_scheme.act.quant(input)
        qweight = quant_scheme.weight.quant(weight)
        if bias is not None:
            bias = bias.to(torch.bfloat16)

        if quant_scheme.same_input:
            input = qinput
        if quant_scheme.same_weight:
            weight = qweight
        ctx.save_for_backward(qinput, qweight, bias)

        output = torch.nn.functional.conv2d(qinput, qweight, bias, stride, padding, dilation, groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        quant_scheme : QuantScheme = ctx.quant_scheme
        qinput, qweight, bias = ctx.saved_tensors

        # print("grad_output", grad_output.var().item(), grad_output.mean().item())
        if not quant_scheme.same_input:
            qinput = quant_scheme.bact.quant(qinput)
        if not quant_scheme.same_weight:
            qweight = quant_scheme.bweight.quant(qweight)

        qgrad_output = quant_scheme.goact.quant(grad_output)
        
        grad_input = torch.nn.grad.conv2d_input(
            qinput.shape, qweight, qgrad_output, 
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        grad_weight = torch.nn.grad.conv2d_weight(
            qinput, qweight.shape, qgrad_output, 
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        # print("grad_weight", grad_weight.var().item(), grad_weight.mean().item())
        grad_bias = qgrad_output.sum(dim=(0, 2, 3)) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class QuantizedModule():
    def __init__(self, quant_scheme):
        self.quant_scheme = quant_scheme

    @classmethod
    def from_full_precision(self, module, quant_scheme):
        raise NotImplementedError


def add_argparse(parser : argparse.ArgumentParser):
    parser.add_argument("--quant_scheme_json", type=str, default="{}")
    return parser


class QuantMethod:
    def __init__(self, number_type, round_mode, number_impl):
        self.number_type = number_type
        self.round_mode = round_mode
        self.number_impl = number_impl

    def quant(self, x):
        if isinstance(self.number_impl, qtorch.FloatingPoint):
            if self.number_impl.exp == 8 and self.number_impl.man == 23:
                return x
        # print("before", x.var(), x.abs().max())
        if SCALING:
            x_scale = x.abs().max()
            x = x / x_scale
        # print("after", x.var(), x.abs().max())
        save_dtype = x.dtype
        x = x.to(torch.float32)
        if isinstance(self.number_impl, qtorch.FloatingPoint):
            result =  qtorch.quant.float_quantize(x, self.number_impl.exp, self.number_impl.man, self.round_mode)
        elif isinstance(self.number_impl, qtorch.FixedPoint):
            result =  qtorch.quant.fixed_point_quantize(x, self.number_impl.wl, self.number_impl.fl, self.number_impl.clamp, self.number_impl.symmetric, self.round_mode)
        elif isinstance(self.number_impl, qtorch.BlockFloatingPoint):
            result = qtorch.quant.block_quantize(x, self.number_impl.wl, self.number_impl.dim, self.round_mode)
        else:
            raise ValueError("Invalid number format")
        if SCALING:
            result = result * x_scale
        return result.to(save_dtype)

    @classmethod
    def from_json(cls, json_dict: dict):
        number_type = json_dict.get("number_type", "fp")
        round_mode = json_dict.get("round_mode", "stochastic")
        if number_type == "fp":
            exp = json_dict.get("exp", 8)
            man = json_dict.get("man", 23)
            number_impl = qtorch.FloatingPoint(exp, man)
        elif number_type == "fixed":
            wl = json_dict.get("wl", 8)
            fl = json_dict.get("fl", 16)
            clamp = json_dict.get("clamp", True)
            symmetric = json_dict.get("symmetric", False)
            number_impl = qtorch.FixedPoint(wl, fl, clamp, symmetric)
        elif number_type == "block":
            wl = json_dict.get("wl", 8)
            dim = json_dict.get("dim", 8)
            number_impl = qtorch.BlockFloatingPoint(wl, dim)
        else:
            raise ValueError("Invalid number format")
        return cls(number_type, round_mode, number_impl)

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
        quant_scheme_json = args.quant_scheme_json
        quant_scheme_json = json.loads(quant_scheme_json)
        return cls.from_json(quant_scheme_json)

    @classmethod
    def from_json(cls, json_dict: dict):
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

FP32 = QuantMethod("fp", "nearest", qtorch.FloatingPoint(8, 23))
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

    def apply_quant_scheme(self, quant_scheme):
        for name, module in self.named_modules():
            if hasattr(module, "quant_scheme"):
                module.quant_scheme = quant_scheme
        return self

    def forward(self, *args, **kw):
        return self.module(*args, **kw)

    def loss_acc(self, X, y):
        return self.module.loss_acc(X, y)
    
    def fix_quant(self, weight_number):
        full_precision_number = qtorch.FloatingPoint(8, 23)
        quant_scheme = QuantScheme(
            fnumber=full_precision_number,
            bnumber=full_precision_number,
            wnumber=weight_number,
            fround_mode="nearest",
            bround_mode="nearest",
            wround_mode="nearest",
            same_input=True,
            same_weight=True
        )
        self.apply_quant_scheme(quant_scheme)
        return self
    
    # def __getattr__(self, name):
    #     try:
    #         return object.__getattribute__(self.module, name)
    #     except AttributeError:
    #         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # def __setattr__(self, name, value):
    #     if name == "module":
    #         self.__dict__["module"] = value
    #     else:
    #         setattr(self.module, name, value)

    # def __hasattr__(self, name):
    #     return hasattr(self.module, name


class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, quant_scheme:"QuantScheme" = None):
        super(QuantLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.quant_scheme = quant_scheme

    def forward(self, input):
        return quant_linear.apply(input, self.weight, self.bias, self.quant_scheme)

    @classmethod
    def from_full_precision(self, module, quant_scheme):
        l = QuantLinear(module.in_features, module.out_features, module.bias is not None, module.weight.device, module.weight.dtype, quant_scheme)
        l.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            l.bias.data.copy_(module.bias.data)
        return l

class QuantConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', 
                 device=None, dtype=None, quant_scheme:"QuantScheme" = None):
        super(QuantConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.quant_scheme = quant_scheme

    def forward(self, input):
        return quant_conv1d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
                                  self.quant_scheme)
    
    @classmethod
    def from_full_precision(self, module, quant_scheme):
        l = QuantConv1d(module.in_channels, module.out_channels, module.kernel_size, 
                        module.stride, module.padding, module.dilation, 
                        module.groups, module.bias is not None, module.padding_mode,
                        module.weight.device, module.weight.dtype, quant_scheme=quant_scheme)
        l.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            l.bias.data.copy_(module.bias.data)
        return l

class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', 
                 device=None, dtype=None, quant_scheme:"QuantScheme" = None):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.quant_scheme = quant_scheme

    def forward(self, input):
        return quant_conv2d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
                                    self.quant_scheme)

    @classmethod
    def from_full_precision(self, module, quant_scheme):
        l = QuantConv2d(module.in_channels, module.out_channels, module.kernel_size, 
                        module.stride, module.padding, module.dilation, 
                        module.groups, module.bias is not None, module.padding_mode,
                        module.weight.device, module.weight.dtype, quant_scheme=quant_scheme)
        l.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            l.bias.data.copy_(module.bias.data)
        return l


def apply_quant_scheme(network, quant_scheme):
    for name, module in network.named_children():
        if hasattr(module, "quant_scheme"):
            module.quant_scheme = quant_scheme
        else:
            apply_quant_scheme(module, quant_scheme)
    return network

def replace_with_quantized(network, quant_scheme):
    to_replace = []
    for name, module in network.named_children():
        if isinstance(module, nn.Linear):
            new_module = QuantLinear.from_full_precision(module, quant_scheme)
            to_replace.append((name, new_module))
        elif isinstance(module, nn.Conv2d):
            new_module = QuantConv2d.from_full_precision(module, quant_scheme)
            to_replace.append((name, new_module))
        elif isinstance(module, nn.Conv1d):
            new_module = QuantConv1d.from_full_precision(module, quant_scheme)
            to_replace.append((name, new_module))
        else:
            replace_with_quantized(module, quant_scheme)
    
    for name, new_module in to_replace:
        setattr(network, name, new_module)

    return network

# class MasterWeightOptimizerWrapper():
#     def __init__(
#             self,
#             master_weight,
#             model_weight,
#             optimizer,
#             scheduler,
#             weight.quant=None,
#             grad_clip=float("inf"),
#             grad_scaling=1.0,
#             grad_stats=False,
#             grad_acc_steps=1
#     ):
#         self.master_weight = master_weight
#         self.model_weight = model_weight
#         self.optimizer = optimizer
#         self.grad_scaling = grad_scaling
#         self.grad_clip = grad_clip
#         self.scheduler = scheduler
#         if weight.quant is None:
#             weight.quant = lambda x: x
#         self.weight.quant = weight.quant
#         self.grad_stats = grad_stats
#         self.grad_acc_steps = grad_acc_steps

#     # --- for mix precision training ---
#     def model_grads_to_master_grads(self):
#         for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
#             if master.grad is None:
#                 master.grad = torch.zeros_like(master.data)
#             assert_nan(model.grad.data)
#             assert_nan(master.grad.data)
#             master.grad.data.copy_(model.grad.data)

#     def master_grad_apply(self, fn):
#         for master in (self.master_weight.parameters()):
#             if master.grad is None:
#                 master.grad = torch.zeros_like(master.data)
#             master.grad.data = fn(master.grad.data)

#     def master_params_to_model_params(self, quantize=True):
#         for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
#             assert_nan(master.data)
#             if quantize:
#                 model.data.copy_(self.weight.quant(master.data))
#             else:
#                 model.data.copy_(master.data)

#     def train_on_batch(self, data, target):
#         self.master_params_to_model_params()
#         self.model_weight.zero_grad()
#         self.master_weight.zero_grad()
#         loss_acc = self.model_weight.loss_acc(data, target)
#         loss = loss_acc["loss"]
#         assert_nan(loss)
#         acc = loss_acc["acc"]
#         # loss = loss * self.grad_scaling
#         loss.backward()
#         self.model_grads_to_master_grads()
#         self.master_grad_apply(assert_nan)
#         # self.master_grad_apply(lambda x: x / self.grad_scaling)
#         # grad_norm = nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
#         self.optimizer.step()
#         self.scheduler.step()
#         if isinstance(acc, torch.Tensor):
#             acc = acc.item()
#         return {"loss": loss.item(),  "acc": acc}

#     def compute_true_grad(self, X, y):
#         self.master_weight.zero_grad()
#         self.master_weight.loss_acc(X, y)["loss"].backward()
#         grads = []
#         for p in self.master_weight.parameters():
#             grads.append(p.grad.data.clone().detach())
#         self.master_weight.zero_grad()
#         return grads

#     def collect_grads(self, target="model_weight"):
#         if target == "model_weight":
#             return [p.grad.data.clone().detach() for p in self.model_weight.parameters()]
#         elif target == "master_weight":
#             return [p.grad.data.clone().detach() for p in self.master_weight.parameters()]

#     def train_compare_true_gradient(self, data, target, X_train, y_train):
#         true_grads = self.compute_true_grad(X_train, y_train)
#         self.master_params_to_model_params()
#         self.model_weight.zero_grad()
#         self.master_weight.zero_grad()
#         loss_acc = self.model_weight.loss_acc(data, target)
#         loss = loss_acc["loss"]
#         assert_nan(loss)
#         acc = loss_acc["acc"]
#         loss.backward()
#         model_grads = self.collect_grads("model_weight")
#         diff = []
#         for g1, g2 in zip(model_grads, true_grads):
#             diff.append(((g1 - g2) ** 2).sum().item())
#         self.model_grads_to_master_grads()
#         self.master_grad_apply(assert_nan)
#         self.optimizer.step()
#         self.scheduler.step()
#         if isinstance(acc, torch.Tensor):
#             acc = acc.item()
#         grad_diff = numpy.sqrt(numpy.mean(diff))
#         return {"loss": loss.item(),  "acc": acc, "grad_diff": grad_diff}

#     def train_with_repeat(self, data, target):
#         self.master_weight.zero_grad()
#         self.model_weight.zero_grad()
#         losses = []
#         acces = []
#         self.grad = {}
#         data = data.repeat(self.grad_acc_steps, 1)
#         target = target.repeat(self.grad_acc_steps, 1)
#         self.master_params_to_model_params()
#         loss_acc = self.model_weight.loss_acc(data, target)
#         loss = loss_acc["loss"]
#         loss.backward()
#         losses.append(loss.item())
#         acc = loss_acc["acc"]
#         if isinstance(acc, torch.Tensor):
#             acc = acc.item()
#         acces.append(acc)
#         self.model_grads_to_master_grads()
#         self.master_grad_apply(lambda x: x / self.grad_acc_steps)
#         # grad_norm = nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
#         self.optimizer.step()
#         self.scheduler.step()
#         return {"loss": numpy.mean(losses),  "acc": numpy.mean(acces)}    

#     def train_with_grad_acc(self, data, target):
#         self.master_weight.zero_grad()
#         self.model_weight.zero_grad()
#         ds = torch.chunk(data, self.grad_acc_steps)
#         ts = torch.chunk(target, self.grad_acc_steps)
#         losses = []
#         acces = []
#         self.grad = {}
#         for d, t in zip(ds, ts):
#             self.master_params_to_model_params()
#             loss_acc = self.model_weight.loss_acc(d, t)
#             loss = loss_acc["loss"]
#             loss.backward()
#             losses.append(loss.item())
#             acc = loss_acc["acc"]
#             if isinstance(acc, torch.Tensor):
#                 acc = acc.item()
#             acces.append(acc)
#         self.model_grads_to_master_grads()
#         self.master_grad_apply(lambda x: x / self.grad_scaling / self.grad_acc_steps)
#         # grad_norm = nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
#         self.optimizer.step()
#         self.scheduler.step()
#         return {"loss": numpy.mean(losses),  "acc": numpy.mean(acces)}


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
