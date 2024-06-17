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

class QuantisedWrapper(nn.Module):
    def __init__(self, network, fnumber, bnumber, fround_mode, bround_mode):
        super(QuantisedWrapper, self).__init__()
        self.network = network
        self.fnumber = fnumber
        self.bnumber = bnumber
        self.fround_mode = fround_mode
        self.bround_mode = bround_mode
        self.quant = Quantizer(forward_number=fnumber, forward_rounding=fround_mode)
        self.quant_b = Quantizer(backward_number=bnumber, backward_rounding=bround_mode)

    def set_number_format(self, *, fnumber, bnumber, fround_mode, bround_mode):
        self.fnumber = fnumber
        self.bnumber = bnumber
        self.fround_mode = fround_mode
        self.bround_mode = bround_mode
        self.quant = Quantizer(forward_number=fnumber, forward_rounding=fround_mode)
        self.quant_b = Quantizer(backward_number=bnumber, backward_rounding=bround_mode)

    def is_full_precision(self):
        if not (isinstance(self.fnumber, qtorch.FloatingPoint) and isinstance(self.bnumber, qtorch.FloatingPoint)):
            return False
        if self.fnumber.exp != 8 or self.bnumber.exp != 8:
            return False
        if self.fnumber.man != 23 or self.bnumber.man != 23:
            return False
        return True


    def forward(self, x):
        if self.is_full_precision():
            return self.network(x)
        assert_nan(x)
        before = self.quant(x)
        assert_nan(before)
        a = self.network(before)
        assert_nan(a)
        after = self.quant_b(a)
        assert_nan(after)
        return after

def apply_number_format(network, fnumber, bnumber, fround_mode, bround_mode):
    for name, module in network.named_children():
        if isinstance(module, QuantisedWrapper):
            module.set_number_format(
                fnumber=fnumber,
                bnumber=bnumber,
                fround_mode=fround_mode,
                bround_mode=bround_mode
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
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layer = QuantisedWrapper(module, fnumber, bnumber, round_mode, round_mode)
            to_replace.append((name, layer))
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
