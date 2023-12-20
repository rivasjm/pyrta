import torch
import numpy as np

print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
device = torch.device("mps")

a = np.array([1, 2, 3, 4]).reshape(2, 2)
ta = torch.from_numpy(a).float().to(device)

b = np.array([5])
tb = torch.from_numpy(b).float().to(device)

print(ta*tb)
print(ta@ta)
print(type(ta))