#%%

from .quant import PerSampleScaledIntQuant
import torch

#%%
def test_single_sample_quantization():
    quant = PerSampleScaledIntQuant(fl=4)
    x = torch.linspace(-10, 10, 1000)   
    y = quant._quant(x).unique()
    print(y)
    expectation = torch.tensor([-10.0000,  -9.3750,  -8.7500,  -8.1250,  -7.5000,  -6.8750,  -6.2500,
         -5.6250,  -5.0000,  -4.3750,  -3.7500,  -3.1250,  -2.5000,  -1.8750,
         -1.2500,  -0.6250,   0.0000,   0.6250,   1.2500,   1.8750,   2.5000,
          3.1250,   3.7500,   4.3750,   5.0000,   5.6250,   6.2500,   6.8750,
          7.5000,   8.1250,   8.7500,   9.3750])
    assert torch.allclose(y, expectation)

def test_different_range():
    quant = PerSampleScaledIntQuant(fl=4)
    x = torch.linspace(-20, 20, 1000)
    y = quant._quant(x).unique()
    expectation = torch.tensor([-20.0000, -18.7500, -17.5000, -16.2500, -15.0000, -13.7500, -12.5000,
        -11.2500, -10.0000,  -8.7500,  -7.5000,  -6.2500,  -5.0000,  -3.7500,
         -2.5000,  -1.2500,  -0.0000,   1.2500,   2.5000,   3.7500,   5.0000,
          6.2500,   7.5000,   8.7500,  10.0000,  11.2500,  12.5000,  13.7500,
         15.0000,  16.2500,  17.5000,  18.7500])

    assert torch.allclose(y, expectation)

def test_batch_quantization():
    quant = PerSampleScaledIntQuant(fl=4, round_mode="nearest")
    x1 = torch.linspace(-10, 10, 1000)
    x2 = torch.linspace(-20, 20, 1000)
    x_batch = torch.vstack([x1, x2])
    y_batch = quant._quant(x_batch)

    y1 = quant._quant(x1)
    y2 = quant._quant(x2)

    expectation = torch.vstack([y1, y2])
    
    assert torch.allclose(y_batch, expectation)

def test_zero_handling():
    quant = PerSampleScaledIntQuant(fl=4)
    x_zero = torch.zeros(100)
    y_zero = quant._quant(x_zero)
    print(y_zero)
    assert torch.all(y_zero == 0)

def test_small_values():
    quant = PerSampleScaledIntQuant(fl=4)
    x_small = torch.linspace(-0.1, 0.1, 100)
    y_small = quant._quant(x_small)
    assert torch.all(y_small >= -1.0)
    assert torch.all(y_small <= 1.0)
    assert len(y_small.unique()) <= 2**5

# Run all tests
if __name__ == "__main__":
    test_single_sample_quantization()
    test_different_range()
    test_batch_quantization()
    test_zero_handling()
    test_small_values()
    print("All tests passed!")
