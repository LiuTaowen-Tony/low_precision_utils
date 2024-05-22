import qtorch
from qtorch.quant import Quantizer
import qtorch.quant
import torch
import numpy


class QuantisedWrapper(nn.Module):
    def __init__(self, network, fnumber, bnumber, fround_mode, bround_mode):
        super(QuantisedWrapper, self).__init__()
        self.network = network
        self.fnumber = fnumber
        self.bnubmer = bnumber
        self.fround_mode = fround_mode
        self.bround_mode = bround_mode
        self.quant = Quantizer(forward_number=fnumber, forward_rounding=fround_mode)
        self.quant_b = Quantizer(backward_number=bnumber, backward_rounding=bround_mode)

    def set_number_format(self, *, fnumber, bnumber, fround_mode, bround_mode):
        self.fnumber = fnumber
        self.bnubmer = bnumber
        self.fround_mode = fround_mode
        self.bround_mode = bround_mode
        self.quant = Quantizer(forward_number=fnumber, forward_rounding=fround_mode)
        self.quant_b = Quantizer(backward_number=bnumber, backward_rounding=bround_mode)

    def forward(self, x):
        if isinstance(self.fnumber, qtorch.FloatingPoint) and isinstance(self.bnumber, qtorch.FloatingPoint):
            if self.fnumber.exp == 8 and self.bnumber.exp == 8:
                if self.fnumber.man == 23 and self.bnumber.man == 23:
                    return self.network(x)
        assert torch.all(x.isnan() == False)
        before = self.quant(x)
        assert torch.all(before.isnan() == False)
        a = self.network(before)
        assert torch.all(a.isnan() == False)
        after = self.quant_b(a)
        return after

def apply_number_format(network, fnumber, bnumber, fround_mode, bround_mode):
    for name, module in network.named_children():
        if isinstance(module, qtorch.quant.QuantisedWrapper):
            module.set_number_format(
                fnumber=fnumber,
                bnumber=bnumber,
                fround_mode=fround_mode,
                bround_mode=bround_mode
            )
        else:
            apply_number_format(module, fnumber, bnumber, fround_mode, bround_mode)

def replace_linear_with_quantized(network, fnumber, bnumber, round_mode):
    # Create a temporary list to store the modules to replace
    to_replace = []
    
    # Iterate to collect modules that need to be replaced
    for name, module in network.named_children():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layer = QuantisedWrapper(module, fnumber, bnumber, round_mode)
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
            weight_quant,
            grad_acc_steps=1,
            grad_clip=float("inf"),
            grad_scaling=1.0,
    ):
        self.master_weight = master_weight
        self.model_weight = model_weight
        self.optimizer = optimizer
        self.grad_scaling = grad_scaling
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.weight_quant = weight_quant
        self.grad_acc_steps = grad_acc_steps
        self.loss_fn = nn.BCELoss()

    # --- for mix precision training ---
    def model_grads_to_master_grads(self):
        for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
            if master.grad is None:
                master.grad = master.data.new(*master.data.size())
                master.grad.data.copy_(model.grad.data)
            else:
                master.grad.data.add_(model.grad.data)

    def master_grad_apply(self, fn):
        for master in (self.master_weight.parameters()):
            if master.grad is None:
                master.grad = fn(master.data.new(*master.data.size()))
            master.grad.data = fn(master.grad.data)

    def master_params_to_model_params(self, quantize=True):
        for model, master in zip(self.model_weight.parameters(), self.master_weight.parameters()):
            if quantize:
                model.data.copy_(self.weight_quant(master.data))
            else:
                model.data.copy_(master.data)


    def train_on_batch(self, data, target):
        self.master_weight.zero_grad()
        self.model_weight.zero_grad()
        output = self.model_weight(data)
        loss = self.loss_fn(output, target)
        # loss = loss * self.grad_scaling
        loss.backward()
        pred = output.round()
        acc = (pred == target).float().mean()
        self.model_grads_to_master_grads()
        # self.master_grad_apply(lambda x: x / self.grad_scaling)
        # grad_norm = nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.master_params_to_model_params()
        return {"loss": loss.item(),  "acc": acc.item()}

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
            output = self.model_weight(d)
            loss = self.loss_fn(output, t)
            loss = loss * self.grad_scaling
            loss.backward()
            acces.append((output.argmax(dim=1) == t).float().mean().item())
            losses.append(loss.item())
        self.model_grads_to_master_grads()
        self.master_grad_apply(lambda x: x / self.grad_scaling / self.grad_acc_steps)
        grad_norm = nn.utils.clip_grad_norm_(self.master_weight.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": numpy.mean(losses),  "acc": numpy.mean(acces), "grad_norm": grad_norm.item()}

