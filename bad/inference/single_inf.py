import cv2
import torch
import numpy as np
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
        self.model.load_state_dict(torch.load(self.path_to_model, map_location=torch.device(self.device)))
        self.model.eval()
       
        x = self.prepare_data()
        with torch.no_grad():
            outputs = self.model(x)

        pred = torch.argmax(outputs, dim=1)
        np_pred = pred.detach().cpu().numpy()
        countour_img = self.remove_artifacts(np_pred[0, :, :])
        return countour_img
    
    
    def remove_artifacts(self, img):
        contour_img = img.astype(np.uint8)
        f_contours = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = f_contours[0] if len(f_contours) == 2 else f_contours[1]
        
        max_ = 0
        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            area = w*h
            if area > max_:
                max_ = area
                
        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            area = w*h
            if area != max_:
                contour_img[y:y+h, x:x+w] = 0
        
        return contour_img
