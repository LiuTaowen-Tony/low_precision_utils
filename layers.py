from torch import nn
from . import functional
from . import quant 

class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, quant_scheme:"quant.QuantScheme" = None):
        super(QuantLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.quant_scheme = quant_scheme

    def forward(self, input):
        return functional.quant_linear.apply(input, self.weight, self.bias, self.quant_scheme)

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
                 device=None, dtype=None, quant_scheme:"quant.QuantScheme" = None):
        super(QuantConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.quant_scheme = quant_scheme

    def forward(self, input):
        return functional.quant_conv1d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
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
                 device=None, dtype=None, quant_scheme:"quant.QuantScheme" = None):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.quant_scheme = quant_scheme

    def forward(self, input):
        return functional.quant_conv2d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
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

