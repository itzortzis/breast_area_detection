import torch
import numpy as np

import training.loaders as loaders

class DPP:
	def __init__(self, dataset):
		print()
		self.dataset = dataset
        
	def run(self):
		self.split_dataset()
		self.normalize_sets()
		self.build_loaders()
        
	def normalize(self, s, max_, min_):
		# max_ = np.max(s[:, :, :, 0])
		# min_ = np.min(s[:, :, :, 0])

		for i in range(len(s)):
			s[i, :, :, 0] = (s[i, :, :, 0] - min_) / (max_ - min_)

		return s


	def normalize_sets(self):
		max_ = np.max(self.train_set[:, :, :, 0])
		min_ = np.min(self.train_set[:, :, :, 0])
		self.train_set = self.normalize(self.train_set, max_, min_)
		self.valid_set = self.normalize(self.valid_set, max_, min_)
		self.set_set = self.normalize(self.test_set, max_, min_)


	def split_dataset(self):
		rp = np.random.permutation(self.dataset.shape[0])
		self.dataset = self.dataset[rp]

		train_set_size = int(0.7 * len(self.dataset))
		valid_set_size = int(0.2 * len(self.dataset))
		test_set_size  = int(0.1 * len(self.dataset))

		train_start = 0
		train_end = train_set_size
		valid_start = train_set_size
		valid_end = valid_start + valid_set_size
		test_start = valid_end
		test_end = test_start + test_set_size

		self.train_set = self.dataset[train_start: train_end, :, :, :]
		self.valid_set = self.dataset[valid_start: valid_end, :, :, :]
		self.test_set = self.dataset[test_start: test_end, :, :, :]


	def build_loaders(self):
		train_set      = loaders.Dataset(self.train_set)
		params         = {'batch_size': 10, 'shuffle': True}
		self.train_ldr = torch.utils.data.DataLoader(train_set, **params)
		valid_set      = loaders.Dataset(self.valid_set)
		params         = {'batch_size': 10, 'shuffle': False}
		self.valid_ldr = torch.utils.data.DataLoader(valid_set, **params)
		test_set       = loaders.Dataset(self.test_set)
		params         = {'batch_size': 10, 'shuffle': False}
		self.test_ldr  = torch.utils.data.DataLoader(test_set, **params)