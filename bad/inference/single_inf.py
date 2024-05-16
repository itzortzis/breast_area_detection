import torch
import bad.utils.models as models


class SingleInf:
    def __init__(self, path_to_inf_model, device, pix_array):
        print()
        self.path_to_model = path_to_inf_model
        self.device = device
        self.pa = pix_array
        self.load_model_struct()
        
        
    def load_model_struct(self):
        in_channels = 1
        out_channels = 2
        self.model = models.UNET(in_channels, out_channels)
        self.model.to(self.device)
    
    
    def prepare_data(self):
        x = torch.from_numpy(self.pa)
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        x = x.to(torch.float32)
        x = x.to(self.device)
        return x
    
    def inference(self):
        self.model.load_state_dict(torch.load(self.path_to_model))
        self.model.eval()
       
        x = self.prepare_data()
        with torch.no_grad():
            outputs = self.model(x)

        pred = torch.argmax(outputs, dim=1)
        np_pred = pred.detach().cpu().numpy()
        return np_pred[0, :, :]
        
        