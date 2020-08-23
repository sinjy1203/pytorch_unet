import numpy as np

a = np.random.rand(28, 28, 1)
a = a.squeeze()
print(a.shape)