import torch
import numpy as np

print("PyTorch version:", torch.__version__)
print("NumPy version:", np.__version__)

# simple tensor test
x = torch.randn(3, 3)
print("Random tensor:")
print(x)
