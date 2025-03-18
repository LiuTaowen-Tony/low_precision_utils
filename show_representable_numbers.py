import torch

from . import quant

def show_representable_numbers(format: quant.QuantMethod, range: float = 1):
    x = torch.linspace(-range, range, 1000)
    y = format.quant(x).unique()
    sorted_y = torch.sort(y).values
    print(sorted_y)


if __name__ == "__main__":
    show_representable_numbers(quant.FPQuant(exp=3, man=2), 10)