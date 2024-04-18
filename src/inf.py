import torch
import numpy as np

from inference.inf import Inf
import utils.loaders as loaders



dataset = np.load("./generated/dataset.npy")



test_set = loaders.Dataset(dataset[1000:, :, :, :])
params = {'batch_size': 50, 'shuffle': False}
test_ldr = torch.utils.data.DataLoader(test_set, **params)

t = Inf('./generated/trained_models/INBreast_bad_0_1000_1713349397.pth', 'cuda', test_ldr)
score = t.inference()
print(score)