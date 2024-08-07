import torch
from torch.autograd import Function
from . import quant

class quant_linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, quant_scheme:"quant.QuantScheme" = None):
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
        quant_scheme : "quant.QuantScheme" = ctx.quant_scheme
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
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, quant_scheme:"quant.QuantScheme"):
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
        quant_scheme : "quant.QuantScheme" = ctx.quant_scheme
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
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, quant_scheme:"quant.QuantScheme"):
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
        quant_scheme : "quant.QuantScheme" = ctx.quant_scheme
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

