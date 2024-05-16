import torch
import numpy as np
from matplotlib import pyplot as plt

from bad.inference.single_inf import SingleInf


sample = np.load("./aux/sample_data/sample.npy")
sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))

# print("Sample shape: ", sample.shape)

t = SingleInf('./aux/model_weights/INBreast_bad_0_1000_1713439049.pth', 'cuda', sample)
pred = t.inference()

# plt.figure()
# plt.imshow(pred, cmap='gray')
# plt.savefig("pred.png")

# plt.figure()
# plt.imshow(sample, cmap='gray')
# plt.savefig("sample.png")
