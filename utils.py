import torch
from torch import nn
import torch.autograd
import torch.nn.grad

import qtorch
import qtorch.quant
import copy

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
    def forward(ctx, input, weight, bias=None, quant_scheme=None):
        ctx.quant_scheme = quant_scheme
        input_shape = input.shape
        # input = input.view(input_shape[0], -1)
        input = input.view(-1, input_shape[-1])
        qinput = quant_scheme.input_quant(input)
        qweight = quant_scheme.weight_quant(weight)

        if quant_scheme.same_input:
            input = qinput
        if quant_scheme.same_weight:
            weight = qweight
        ctx.save_for_backward(qinput, qweight, bias)

        output = qinput.mm(qweight.t())
        if bias is not None:
            output += bias
        return output.view(*input_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        quant_scheme = ctx.quant_scheme
        grad_output_shape = grad_output.shape
        # print(grad_output_shape)
        grad_output = grad_output.reshape(-1, grad_output_shape[-1])
        qinput, qweight, bias = ctx.saved_tensors

        if not quant_scheme.same_input:
            qinput = quant_scheme.back_input_quant(qinput)
        if not quant_scheme.same_weight:
            qweight = quant_scheme.back_weight_quant(qweight)

        qgrad_output = quant_scheme.grad_quant(grad_output)

        grad_input = qgrad_output.mm(qweight)
        grad_weight = qgrad_output.t().mm(qinput)

        grad_bias = qgrad_output.sum(0) if bias is not None else None
        return grad_input.view(*grad_output_shape[:-1], -1), grad_weight, grad_bias, None


class quant_conv1d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, quant_scheme):
        ctx.quant_scheme = quant_scheme
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        qinput = quant_scheme.input_quant(input)
        qweight = quant_scheme.weight_quant(weight)

        if quant_scheme.same_input:
            input = qinput
        if quant_scheme.same_weight:
            weight = qweight
        ctx.save_for_backward(qinput, qweight, bias)

        output = torch.nn.functional.conv1d(qinput, qweight, bias, stride, padding, dilation, groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        quant_scheme = ctx.quant_scheme
        qinput, qweight, bias = ctx.saved_tensors

        if not quant_scheme.same_input:
            qinput = quant_scheme.back_input_quant(qinput)
        if not quant_scheme.same_weight:
            qweight = quant_scheme.back_weight_quant(qweight)

        qgrad_output = quant_scheme.grad_quant(grad_output)
        
        grad_input = torch.nn.grad.conv1d_input(
            qinput.shape, qweight, qgrad_output, 
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        grad_weight = torch.nn.grad.conv1d_weight(
            qinput, qweight.shape, qgrad_output, 
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        grad_bias = qgrad_output.sum(dim=(0, 2)) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class quant_conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, quant_scheme):
        ctx.quant_scheme = quant_scheme
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        qinput = quant_scheme.input_quant(input)
        qweight = quant_scheme.weight_quant(weight)

        if quant_scheme.same_input:
            input = qinput
        if quant_scheme.same_weight:
            weight = qweight
        ctx.save_for_backward(qinput, qweight, bias)

        output = torch.nn.functional.conv2d(qinput, qweight, bias, stride, padding, dilation, groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        quant_scheme = ctx.quant_scheme
        qinput, qweight, bias = ctx.saved_tensors

        if not quant_scheme.same_input:
            qinput = quant_scheme.back_input_quant(qinput)
        if not quant_scheme.same_weight:
            qweight = quant_scheme.back_weight_quant(qweight)

        qgrad_output = quant_scheme.grad_quant(grad_output)
        
        grad_input = torch.nn.grad.conv2d_input(
            qinput.shape, qweight, qgrad_output, 
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        grad_weight = torch.nn.grad.conv2d_weight(
            qinput, qweight.shape, qgrad_output, 
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        grad_bias = qgrad_output.sum(dim=(0, 2, 3)) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class QuantizedModule():
    def __init__(self, quant_scheme):
        self.quant_scheme = quant_scheme

    @classmethod
    def from_full_precision(self, module, quant_scheme):
        raise NotImplementedError

class QuantScheme:
    def __init__(self, fnumber, bnumber, wnumber, 
                 fround_mode="stochastic", 
                 bround_mode="stochastic", 
                 wround_mode="stochastic", 
                 same_input=False,
                 same_weight=False,
                 bfnumber=None,
                 bwnumber=None,
                 bfround_mode=None,
                 bwround_mode=None,):
        self.fnumber = fnumber
        self.bnumber = bnumber
        self.wnumber = wnumber
        self.bfnumber = bfnumber or fnumber
        self.bwnumber = bwnumber or wnumber
        self.fround_mode = fround_mode
        self.bround_mode = bround_mode
        self.wround_mode = wround_mode
        self.bfround_mode = bfround_mode or fround_mode
        self.bwround_mode = bwround_mode or wround_mode
        self.same_input = same_input
        self.same_weight = same_weight

    def quant(self, x, number, round_mode):
        if isinstance(number, qtorch.FloatingPoint):
            if number.exp == 8 and number.man == 23:
                return x
            return qtorch.quant.float_quantize(x, number.exp, number.man, round_mode)
        elif isinstance(number, qtorch.FixedPoint):
            return qtorch.quant.fixed_point_quantize(x, number.wl, number.fl, number.clamp, number.symmetric, round_mode)
        elif isinstance(number, qtorch.BlockFloatingPoint):
            return qtorch.quant.block_quantize(x, number.wl, number.dim, round_mode)
        else:
            raise ValueError("Invalid number format")

    def input_quant(self, x):
        return self.quant(x, self.fnumber, self.fround_mode)
    
    def weight_quant(self, x):
        return self.quant(x, self.wnumber, self.wround_mode)

    def grad_quant(self, x):
        return self.quant(x, self.bnumber, self.bround_mode)

    def back_input_quant(self, x):
        return self.quant(x, self.bfnumber, self.bfround_mode)
    
    def back_weight_quant(self, x):
        return self.quant(x, self.bwnumber, self.bwround_mode)

    def __str__(self):
        return self.__dict__.__str__()


class QuantWrapper(nn.Module):
    def __init__(self, module, quant_scheme):
        super(QuantWrapper, self).__init__()
        module = replace_with_quantized(module, quant_scheme)
        self.module = module

    def apply_quant_scheme(self, quant_scheme):
        for name, module in self.named_modules():
            if hasattr(module, "quant_scheme"):
                module.quant_scheme = quant_scheme
        return self

    def forward(self, *args, **kw):
        return self.module(*args, **kw)
    
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
    
    def __getattr__(self, name):
        try:
            return object.__getattribute__(self.module, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "module":
            self.__dict__["module"] = value
        else:
            setattr(self.module, name, value)

    # def __hasattr__(self, name):
    #     return hasattr(self.module, name


class QuantLinear(nn.Linear, QuantizedModule):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, quant_scheme=None):
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

class QuantConv1d(nn.Conv1d, QuantizedModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', 
                 device=None, dtype=None, quant_scheme=None):
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

class QuantConv2d(nn.Conv2d, QuantizedModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', 
                 device=None, dtype=None, quant_scheme=None):
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
#             weight_quant=None,
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
#         if weight_quant is None:
#             weight_quant = lambda x: x
#         self.weight_quant = weight_quant
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
#                 model.data.copy_(self.weight_quant(master.data))
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
