import numpy as np
import os
import matplotlib.pyplot as plt

result_dir = "./result/numpy"

lst_data = os.listdir(result_dir)

lst_input = [f for f in lst_data if f.startswith("input")]
lst_label = [f for f in lst_data if f.startswith("label")]
lst_output = [f for f in lst_data if f.startswith('output')]

idx = 0

input = np.load(os.path.join(result_dir, lst_input[idx]))
output = np.load(os.path.join(result_dir, lst_output[idx]))
label = np.load(os.path.join(result_dir, lst_label[idx]))

plt.subplot(131)
plt.imshow(input, cmap='gray')
plt.title("input")

plt.subplot(132)
plt.imshow(output, cmap='gray')
plt.title("output")

plt.subplot(133)
plt.imshow(label, cmap='gray')
plt.title("label")

plt.show()