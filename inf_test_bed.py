import torch
import numpy as np

from bad.inference.inf import Inf
import bad.utils.loaders as loaders


samples = np.load("./aux/sample_data/sample_batch.npy")

print(np.min(samples), np.max(samples))

samples[:, :, :, 0] = (samples[:, :, :, 0] - np.min(samples[:, :, :, 0])) / (np.max(samples[:, :, :, 0]) - np.min(samples[:, :, :, 0]))

test_set = loaders.Dataset(samples[:, :, :, :])
params = {'batch_size': 10, 'shuffle': False}
test_ldr = torch.utils.data.DataLoader(test_set, **params)

t = Inf('./aux/model_weights/INBreast_bad_0_1000_1713439049.pth', 'cuda', test_ldr)
score = t.inference()
print(score)