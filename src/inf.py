import torch
import numpy as np

from inference.inf import Inf
import utils.loaders as loaders

import sys
print(sys.path)

dataset = np.load("./generated/dataset.npy")

print(np.min(dataset), np.max(dataset))
temp = dataset[1000:, :, :, 0]
dataset[1000:, :, :, 0] = (dataset[1000:, :, :, 0] - np.min(dataset[1000:, :, :, 0])) / (np.max(dataset[1000:, :, :, 0]) - np.min(dataset[1000:, :, :, 0]))

test_set = loaders.Dataset(dataset[1000:, :, :, :])
params = {'batch_size': 10, 'shuffle': False}
test_ldr = torch.utils.data.DataLoader(test_set, **params)

t = Inf('./generated/trained_models/INBreast_bad_0_1000_1713439049.pth', 'cuda', test_ldr)
score = t.inference()
print(score)