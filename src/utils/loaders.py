import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
	def __init__(self, d):
		self.dtst = d

	def __len__(self):
		return len(self.dtst)

	def __getitem__(self, index):
		obj = self.dtst[index, :, :, :]
	
		x = torch.from_numpy(obj[:, :, 0])
		y = torch.from_numpy(obj[:, :, 1])
		
		
		return x, y