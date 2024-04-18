import torch
from torchmetrics import F1Score
from matplotlib import pyplot as plt


import utils.models as models


class Inf:
    def __init__(self, path_to_inf_model, device, set_ldr):
        print()
        self.path_to_model = path_to_inf_model
        self.device = device
        self.set_ldr = set_ldr
        self.load_model_struct()
        
        
    def load_model_struct(self):
        in_channels = 1
        out_channels = 2
        self.model = models.UNET(in_channels, out_channels)
        self.model.to(self.device)
    
    
    def prepare_data(self, x, y):
        if len(x.size()) < 4:
            x = torch.unsqueeze(x, 1)
        else:
            x = x.movedim(2, -1)
            x = x.movedim(1, 2)

        x = x.to(torch.float32)
        y = y.to(torch.int64)

        x = x.to(self.device)
        y = y.to(self.device)

        return x, y
    
    
    def inference(self):
        self.model.load_state_dict(torch.load(self.path_to_model))
        self.model.eval()
        current_score = 0.0
        self.metric = F1Score(task="binary", num_classes=2)
        self.metric.to(self.device)
        idx = 0
        for x, y in self.set_ldr:
            x,  y = self.prepare_data(x, y)

            with torch.no_grad():
                outputs = self.model(x)

            preds = torch.argmax(outputs, dim=1)
            # print(preds.size(), y.size(), idx)
            self.show_imgs(y, preds, idx)
            self.metric.update(preds, y)
            idx += 1

        test_set_score = self.metric.compute()
        self.metric.reset()
        return test_set_score.item()
    
    
    def show_imgs(self, y, preds, idx):
        np_y = y.detach().cpu().numpy()
        np_preds = preds.detach().cpu().numpy()
        
        mask = np_y[2, :, :]
        pred = np_preds[2, :, :]
        
        fig = plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred, cmap='gray')
        plt.axis('off')
        
        plt.savefig('./generated/preds/' + str(idx) + '.png')
        plt.close()
        
        