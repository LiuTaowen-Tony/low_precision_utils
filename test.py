import torch
import torch.nn as nn
import torch.autograd as autograd
import unittest
from utils import *


class TestQuantization(unittest.TestCase):
    def setUp(self):
        self.input_linear = torch.randn(10, 5, requires_grad=True)
        self.weight_linear = torch.randn(3, 5, requires_grad=True)
        self.bias_linear = torch.randn(3, requires_grad=True)
        
        self.input_conv2d = torch.randn(1, 3, 8, 8, requires_grad=True)
        self.weight_conv2d = torch.randn(6, 3, 3, 3, requires_grad=True)
        self.bias_conv2d = torch.randn(6, requires_grad=True)
        
        self.quant_scheme = QuantScheme(
            fnumber=qtorch.FloatingPoint(8, 1),
            bnumber=qtorch.FloatingPoint(8, 1),
            wnumber=qtorch.FloatingPoint(8, 1),
        )
    
    def test_quant_linear_forward_values(self):
        result = quant_linear.apply(self.input_linear, self.weight_linear, self.bias_linear, self.quant_scheme)
        expected_output = (self.input_linear * 0.1).mm((self.weight_linear * 0.1).t())
        expected_output += self.bias_linear
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-5))
    
    def test_quant_linear_backward_values(self):
        output = quant_linear.apply(self.input_linear, self.weight_linear, self.bias_linear, self.quant_scheme)
        loss = output.sum()
        loss.backward()
        expected_grad_input = (self.weight_linear * 0.1).t().mm(torch.ones_like(output) * 0.1)
        expected_grad_weight = torch.ones_like(output).t().mm(self.input_linear * 0.1)
        self.assertTrue(torch.allclose(self.input_linear.grad, expected_grad_input, atol=1e-5))
        self.assertTrue(torch.allclose(self.weight_linear.grad, expected_grad_weight, atol=1e-5))
        self.assertIsNotNone(self.bias_linear.grad)
    
    def test_quant_conv2d_forward_values(self):
        result = quant_conv2d.apply(self.input_conv2d, self.weight_conv2d, self.bias_conv2d, 
                                    stride=1, padding=1, dilation=1, groups=1, quant_scheme=self.quant_scheme)
        expected_output = torch.nn.functional.conv2d(self.input_conv2d * 0.1, self.weight_conv2d * 0.1, 
                                                     self.bias_conv2d, stride=1, padding=1, dilation=1, groups=1)
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-5))
    
    def test_quant_conv2d_backward_values(self):
        output = quant_conv2d.apply(self.input_conv2d, self.weight_conv2d, self.bias_conv2d, 
                                    stride=1, padding=1, dilation=1, groups=1, quant_scheme=self.quant_scheme)
        loss = output.sum()
        loss.backward()
        expected_grad_input = torch.nn.grad.conv2d_input(
            self.input_conv2d.shape, self.weight_conv2d * 0.1, torch.ones_like(output) * 0.1, 
            stride=1, padding=1, dilation=1, groups=1)
        expected_grad_weight = torch.nn.grad.conv2d_weight(
            self.input_conv2d * 0.1, self.weight_conv2d.shape, torch.ones_like(output) * 0.1, 
            stride=1, padding=1, dilation=1, groups=1)
        self.assertTrue(torch.allclose(self.input_conv2d.grad, expected_grad_input, atol=1e-5))
        self.assertTrue(torch.allclose(self.weight_conv2d.grad, expected_grad_weight, atol=1e-5))
        self.assertIsNotNone(self.bias_conv2d.grad)
    
    def test_quant_linear_module_values(self):
        model = QuantLinear(5, 3, quant_scheme=self.quant_scheme)
        model.weight.data = self.weight_linear.data.clone()
        model.bias.data = self.bias_linear.data.clone()
        output = model(self.input_linear)
        expected_output = (self.input_linear * 0.1).mm((self.weight_linear * 0.1).t())
        expected_output += self.bias_linear
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-5))
    
    def test_quant_conv2d_module_values(self):
        model = QuantConv2d(3, 6, 3, padding=1, quant_scheme=self.quant_scheme)
        model.weight.data = self.weight_conv2d.data.clone()
        model.bias.data = self.bias_conv2d.data.clone()
        output = model(self.input_conv2d)
        expected_output = torch.nn.functional.conv2d(self.input_conv2d * 0.1, self.weight_conv2d * 0.1, 
                                                     self.bias_conv2d, stride=1, padding=1, dilation=1, groups=1)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
