
import sys
import torch
import numpy as np


from utils.training_pipeline import Training
import utils.models as models

import utils.loaders as loaders

paths = {
	'dataset': './generated/dataset.npy'
}



if __name__ == "__main__":

    dataset = np.load(paths['dataset'])
    print(dataset.shape)
    in_channels = 1
    out_channels = 2
    unet = models.UNET(in_channels, out_channels)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=0.01)

    comps = {
        'model': unet,
        'opt': opt,
        'loss_fn': loss_fn,
        'dataset': dataset[:1000, :, :, :]
    }
    print(sys.argv)
    params = {
        'epochs': 100,
        'dtst_name': 'INBreast_bad_0_1000',
        'epoch_thresh': 40,
        'score_thresh': 0.75,
        'device': 'cuda',
        'batch_size': 10,
        'inf_model_name': 'INBreast_bad_0_1000_1713349397.pth',
    }
    tr_paths = {
        'trained_models': './generated/trained_models/',
        'metrics': './generated/metrics/',
        'figures': './generated/figures/'
    }
    t = Training(comps, params, tr_paths)
    # t.main_training()
    
    
    test_set = loaders.Dataset(dataset[1000:, :, :, :])
    params = {'batch_size': 10, 'shuffle': False}
    test_ldr = torch.utils.data.DataLoader(test_set, **params)
    
    score = t.ext_inference(test_ldr)
    print(score)