import torch
from torch.autograd import Function
from . import quant

def quant_forward(ctx, input, weight, bias, quant_scheme:"quant.QuantScheme"):
    ctx.quant_scheme = quant_scheme
    qinput = quant_scheme.act.quant(input)
    qweight = quant_scheme.weight.quant(weight)
    if quant_scheme.same_input:
        input = qinput
    if quant_scheme.same_weight:
        weight = qweight
    ctx.save_for_backward(input, weight, bias)
    return qinput, qweight

def quant_backward(ctx, grad_output):
    quant_scheme : "quant.QuantScheme" = ctx.quant_scheme
    qinput, qweight, bias = ctx.saved_tensors

    if not quant_scheme.same_input:
        qinput = quant_scheme.bact.quant(qinput)
    if not quant_scheme.same_weight:
        qweight = quant_scheme.bweight.quant(qweight)

    qgrad_for_act = quant_scheme.goact.quant(grad_output)
    qgrad_for_weight = quant_scheme.goweight.quant(grad_output)
    return qinput, qweight, qgrad_for_act, qgrad_for_weight, bias

class quant_linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, quant_scheme:"quant.QuantScheme" = None):
        input_shape = input.shape
        input = input.view(-1, input_shape[-1])
        qinput, qweight = quant_forward(ctx, input, weight, bias, quant_scheme)
        output = qinput.mm(qweight.t())
        if bias is not None:
            output += bias
        return output.view(*input_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_shape = grad_output.shape
        grad_output = grad_output.view(-1, grad_output_shape[-1])
        qinput, qweight, qgrad_for_input, qgrad_for_weight, bias = quant_backward(ctx, grad_output)

        grad_input = qgrad_for_input.mm(qweight)
        grad_weight = qgrad_for_weight.t().mm(qinput)

        grad_bias = grad_output.sum(0) if bias is not None else None
        return grad_input.view(*grad_output_shape[:-1], -1), grad_weight, grad_bias, None

class quant_conv1d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, quant_scheme:"quant.QuantScheme"):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        qinput, qweight = quant_forward(ctx, input, weight, bias, quant_scheme)

        output = torch.nn.functional.conv1d(qinput, qweight, bias, stride, padding, dilation, groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        qinput, qweight, qgrad_for_input, qgrad_for_weight, bias = quant_backward(ctx, grad_output)

        
        grad_input = torch.nn.grad.conv1d_input(
            qinput.shape, qweight, qgrad_for_input, 
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        grad_weight = torch.nn.grad.conv1d_weight(
            qinput, qweight.shape, qgrad_for_weight, 
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        grad_bias = grad_output.sum(dim=(0, 2)) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class quant_conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, quant_scheme:"quant.QuantScheme"):
        ctx.quant_scheme = quant_scheme
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        qinput, qweight = quant_forward(ctx, input, weight, bias, quant_scheme)

        output = torch.nn.functional.conv2d(qinput, qweight, bias, stride, padding, dilation, groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        qinput, qweight, qgrad_for_input, qgrad_for_weight, bias = quant_backward(ctx, grad_output)
        
        grad_input = torch.nn.grad.conv2d_input(
            qinput.shape, qweight, qgrad_for_input, 
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        grad_weight = torch.nn.grad.conv2d_weight(
            qinput, qweight.shape, qgrad_for_weight, 
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        # print("grad_weight", grad_weight.var().item(), grad_weight.mean().item())
        grad_bias = qgrad_for_input.sum(dim=(0, 2, 3)) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

