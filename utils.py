import torch.autograd
import torch
from torch import nn
import qtorch
from qtorch.quant import Quantizer
import qtorch.quant
import copy
import numpy
from torch import nn

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
    def forward(ctx, input, weight, bias=None, fnumber=None, fround_mode="nearest", bnumber=None, bround_mode="nearest", same_input=False):
        # Save tensors for backward
        qinput = qtorch.quant.quant_function.float_quantize(input, fnumber.exp, fnumber.man, fround_mode)
        if same_input:
            input = qinput

        tensors_to_save = [input, weight]
        if bias is not None:
            tensors_to_save.append(bias)
        ctx.save_for_backward(*tensors_to_save)

        # Save non-tensor objects as attributes
        ctx.bnumber = bnumber
        ctx.bround_mode = bround_mode
        ctx.fnumber = fnumber
        ctx.fround_mode = fround_mode
        ctx.same_input = same_input

        # Perform forward pass
        output = qinput.mm(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, *bias = ctx.saved_tensors

        # Retrieve non-tensor objects from context
        bnumber = ctx.bnumber
        bround_mode = ctx.bround_mode
        fnumber = ctx.fnumber
        fround_mode = ctx.fround_mode

        # Perform quantization in backward pass
        if not ctx.same_input:
            input = qtorch.quant.quant_function.float_quantize(input, fnumber.exp, fnumber.man, fround_mode)
        grad_output = qtorch.quant.quant_function.float_quantize(grad_output, bnumber.exp, bnumber.man, bround_mode)

        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)

        if bias:
            grad_bias = grad_output.sum(0)
            return grad_input, grad_weight, grad_bias, None, None, None, None, None
        else:
            return grad_input, grad_weight, None, None, None, None, None, None


class quant_conv2d(Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None, fnumber=None, fround_mode="nearest", bnumber=None, bround_mode="nearest"):
        # Ensure quantization parameters exist
        if not (hasattr(fnumber, 'exp') and hasattr(fnumber, 'man')):
            raise AttributeError("fnumber must have 'exp' and 'man' attributes.")
        if not (hasattr(bnumber, 'exp') and hasattr(bnumber, 'man')):
            raise AttributeError("bnumber must have 'exp' and 'man' attributes.")
        
        # Save tensors for backward
        tensors_to_save = [input, weight]
        if bias is not None:
            tensors_to_save.append(bias)
        ctx.save_for_backward(*tensors_to_save)

        # Save non-tensor objects as attributes
        ctx.fnumber = fnumber
        ctx.fround_mode = fround_mode
        ctx.bnumber = bnumber
        ctx.bround_mode = bround_mode

        # Perform forward pass with quantization
        output = torch.nn.functional.conv2d(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, *bias = ctx.saved_tensors

        # Retrieve non-tensor objects from context
        fnumber = ctx.fnumber
        fround_mode = ctx.fround_mode
        bnumber = ctx.bnumber
        bround_mode = ctx.bround_mode

        # Quantize grad_output
        grad_output = qtorch.quant.float_quantize(grad_output, bnumber.exp, bnumber.man, bround_mode)
        
        # Compute gradients
        grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output)
        grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output)
        grad_bias = grad_output.sum((0, 2, 3)) if bias else None

        # Return gradients for all inputs in the order they were received by the forward method
        return grad_input, grad_weight, grad_bias, None, None, None, None


class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, fnumber=None, bnumber=None, fround_mode="nearest", bround_mode="nearest", same_input=False):
        super(QuantLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.fnumber = fnumber
        self.bnumber = bnumber
        self.fround_mode = fround_mode
        self.bround_mode = bround_mode
        self.same_input = same_input

    def forward(self, input):
        if self.fnumber is None and self.bnumber is None:
            return super(QuantLinear, self).forward(input)
        if self.fnumber.man == 23 and self.fnumber.exp == 8 and self.bnumber.exp == 8 and self.bnumber.man == 23:
            return super(QuantLinear, self).forward(input)
        return quant_linear.apply(input, self.weight, self.bias, self.fnumber, self.fround_mode, self.bnumber, self.bround_mode, self.same_input)

    def set_number_format(self, *, fnumber, bnumber, fround_mode, bround_mode, same_input):
        self.fnumber = fnumber
        self.bnumber = bnumber
        self.fround_mode = fround_mode
        self.bround_mode = bround_mode
        self.same_input = same_input

    @classmethod
    def from_full_precision(self, module, fnumber=None, bnumber=None, fround_mode="nearest", bround_mode="nearest"):
        l = QuantLinear(module.in_features, module.out_features, module.bias is not None, module.weight.device, module.weight.dtype,
                        fnumber=fnumber, bnumber=bnumber, fround_mode=fround_mode, bround_mode=bround_mode)
        l.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            l.bias.data.copy_(module.bias.data)
        return l

class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='replicate', device=None, dtype=None, fnumber=None, bnumber=None, fround_mode="nearest", bround_mode="nearest"):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.fnumber = fnumber
        self.bnumber = bnumber
        self.fround_mode = fround_mode
        self.bround_mode = bround_mode

    def forward(self, input):
        if self.fnumber is None and self.bnumber is None:
            return super(QuantConv2d, self).forward(input)
        if self.fnumber.man == 23 and self.fnumber.exp == 8 and self.bnumber.exp == 8 and self.bnumber.man == 23:
            return super(QuantConv2d, self).forward(input)
        return quant_conv2d.apply(input, self.weight, self.bias, self.fnumber, self.fround_mode, self.bnumber, self.bround_mode)

    def set_number_format(self, *, fnumber, bnumber, fround_mode, bround_mode, same_input=False):
        self.fnumber = fnumber
        self.bnumber = bnumber
        self.fround_mode = fround_mode
        self.bround_mode = bround_mode
        self.same_input = same_input

    @classmethod
    def from_full_precision(self, module, fnumber=None, bnumber=None, fround_mode="nearest", bround_mode="nearest"):
        l = QuantConv2d(module.in_channels, module.out_channels, module.kernel_size, 
                        module.stride, module.padding, module.dilation, 
                        module.groups, module.bias is not None, module.padding_mode,
                        module.weight.device, module.weight.dtype,
                        fnumber=fnumber, bnumber=bnumber, fround_mode=fround_mode, bround_mode=bround_mode)
        l.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            l.bias.data.copy_(module.bias.data)
        return l


def apply_number_format(network, fnumber, bnumber, fround_mode, bround_mode, same_input=False):
    for name, module in network.named_children():
        if hasattr(module, "set_number_format"):
            module.set_number_format(
                fnumber=fnumber,
                bnumber=bnumber,
                fround_mode=fround_mode,
                bround_mode=bround_mode,
                same_input=same_input
            )
        else:
            apply_number_format(module, fnumber, bnumber, fround_mode, bround_mode)
    return network

def replace_linear_with_quantized(network, fnumber=None, bnumber=None, round_mode=
                                  "stochastic"):
    # Create a temporary list to store the modules to replace
    fnumber = fnumber or qtorch.FloatingPoint(8, 23)
    bnumber = bnumber or qtorch.FloatingPoint(8, 23)

    to_replace = []
    
    # Iterate to collect modules that need to be replaced
    for name, module in network.named_children():
        if isinstance(module, nn.Linear):
            new_module = QuantLinear.from_full_precision(module, fnumber, bnumber, round_mode, round_mode)
            to_replace.append((name, new_module))
        elif isinstance(module, nn.Conv2d):
            new_module = QuantConv2d.from_full_precision(module, fnumber, bnumber, round_mode, round_mode)
            to_replace.append((name, new_module))
        else:
            replace_linear_with_quantized(module, fnumber, bnumber, round_mode)
    
    for name, new_module in to_replace:
        setattr(network, name, new_module)
    
    return network

class MasterWeightOptimizerWrapper():
    def __init__(
            self,
            master_weight,
            model_weight,
            optimizer,
            scheduler,
            weight_quant=None,
            grad_clip=float("inf"),
            grad_scaling=1.0,
            grad_stats=False,
            grad_acc_steps=1
    ):
        self.master_weight = master_weight
        self.model_weight = model_weight
        self.optimizer = optimizer
        self.grad_scaling = grad_scaling
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        if weight_quant is None:
            weight_quant = lambda x: x
        self.weight_quant = weight_quant
        self.grad_stats = grad_stats
        self.grad_acc_steps = grad_acc_steps

    # --- for mix precision training ---
    def model_grads_to_master_grads(self):
        for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
            if master.grad is None:
                master.grad = torch.zeros_like(master.data)
            assert_nan(model.grad.data)
            assert_nan(master.grad.data)
            master.grad.data.copy_(model.grad.data)

    def master_grad_apply(self, fn):
        for master in (self.master_weight.parameters()):
            if master.grad is None:
                master.grad = torch.zeros_like(master.data)
            master.grad.data = fn(master.grad.data)

    def master_params_to_model_params(self, quantize=True):
        for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
            assert_nan(master.data)
            if quantize:
                model.data.copy_(self.weight_quant(master.data))
            else:
                model.data.copy_(master.data)

    def train_on_batch(self, data, target):
        self.master_params_to_model_params()
        self.model_weight.zero_grad()
        self.master_weight.zero_grad()
        loss_acc = self.model_weight.loss_acc(data, target)
        loss = loss_acc["loss"]
        assert_nan(loss)
        acc = loss_acc["acc"]
        # loss = loss * self.grad_scaling
        loss.backward()
        self.model_grads_to_master_grads()
        self.master_grad_apply(assert_nan)
        # self.master_grad_apply(lambda x: x / self.grad_scaling)
        # grad_norm = nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        if isinstance(acc, torch.Tensor):
            acc = acc.item()
        return {"loss": loss.item(),  "acc": acc}

    def compute_true_grad(self, X, y):
        self.master_weight.zero_grad()
        self.master_weight.loss_acc(X, y)["loss"].backward()
        grads = []
        for p in self.master_weight.parameters():
            grads.append(p.grad.data.clone().detach())
        self.master_weight.zero_grad()
        return grads

    def collect_grads(self, target="model_weight"):
        if target == "model_weight":
            return [p.grad.data.clone().detach() for p in self.model_weight.parameters()]
        elif target == "master_weight":
            return [p.grad.data.clone().detach() for p in self.master_weight.parameters()]

    def train_compare_true_gradient(self, data, target, X_train, y_train):
        true_grads = self.compute_true_grad(X_train, y_train)
        self.master_params_to_model_params()
        self.model_weight.zero_grad()
        self.master_weight.zero_grad()
        loss_acc = self.model_weight.loss_acc(data, target)
        loss = loss_acc["loss"]
        assert_nan(loss)
        acc = loss_acc["acc"]
        loss.backward()
        model_grads = self.collect_grads("model_weight")
        diff = []
        for g1, g2 in zip(model_grads, true_grads):
            diff.append(((g1 - g2) ** 2).sum().item())
        self.model_grads_to_master_grads()
        self.master_grad_apply(assert_nan)
        self.optimizer.step()
        self.scheduler.step()
        if isinstance(acc, torch.Tensor):
            acc = acc.item()
        grad_diff = numpy.sqrt(numpy.mean(diff))
        return {"loss": loss.item(),  "acc": acc, "grad_diff": grad_diff}

    def train_with_repeat(self, data, target):
        self.master_weight.zero_grad()
        self.model_weight.zero_grad()
        losses = []
        acces = []
        self.grad = {}
        data = data.repeat(self.grad_acc_steps, 1)
        target = target.repeat(self.grad_acc_steps, 1)
        self.master_params_to_model_params()
        loss_acc = self.model_weight.loss_acc(data, target)
        loss = loss_acc["loss"]
        loss.backward()
        losses.append(loss.item())
        acc = loss_acc["acc"]
        if isinstance(acc, torch.Tensor):
            acc = acc.item()
        acces.append(acc)
        self.model_grads_to_master_grads()
        self.master_grad_apply(lambda x: x / self.grad_acc_steps)
        # grad_norm = nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": numpy.mean(losses),  "acc": numpy.mean(acces)}    

    def train_with_grad_acc(self, data, target):
        self.master_weight.zero_grad()
        self.model_weight.zero_grad()
        ds = torch.chunk(data, self.grad_acc_steps)
        ts = torch.chunk(target, self.grad_acc_steps)
        losses = []
        acces = []
        self.grad = {}
        for d, t in zip(ds, ts):
            self.master_params_to_model_params()
            loss_acc = self.model_weight.loss_acc(d, t)
            loss = loss_acc["loss"]
            loss.backward()
            losses.append(loss.item())
            acc = loss_acc["acc"]
            if isinstance(acc, torch.Tensor):
                acc = acc.item()
            acces.append(acc)
        self.model_grads_to_master_grads()
        self.master_grad_apply(lambda x: x / self.grad_scaling / self.grad_acc_steps)
        # grad_norm = nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": numpy.mean(losses),  "acc": numpy.mean(acces)}


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
