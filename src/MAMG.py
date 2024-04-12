# Mammogram Annotation Mask Generator

import cv2
import random
import numpy as np
import pydicom as pdcm
from sklearn.cluster import KMeans
from skimage.color import label2rgb


class MAMG:
    
    def __init__(self, paths, img_name):
        self.paths = paths
        self.img_name = img_name
        
    def run(self):
        self.get_pixel_array(self.img_name)
        self.get_laterality()
        self.add_random_label()
        self.cluster_pixel_array()
        self.detect_contours()
        self.throw_small_contours()
        
        
    def get_pixel_array(self, img_name):

        path_to_dcm = self.paths['dcm'] + img_name + '.dcm'
        img_obj = pdcm.dcmread(path_to_dcm)
        img = img_obj.pixel_array
        self.pixel_array = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
        
    
    # O corresponds to left 
    # 1 corresponds to right
    def get_laterality(self):
        self.lat = 1 if np.sum(self.pixel_array[:, 20]) == 0 else 0
        
        
    def add_random_label(self):
        x = random.randint(10, 60)
        y = random.randint(150, 200)
        w = random.randint(20, 50)
        h = random.randint(5, 15)
        
        if self.lat == 1:
            y = 256 - y - w
        
        self.pixel_array[x:x+h, y:y+w] = 0
        
        
    def cluster_pixel_array(self):
        img = self.pixel_array.copy()
        norm_img = (img - np.min(img)) / (np.max(img) - np.min(img))
        flat_img = np.reshape(norm_img, [-1, 1])
        cm = KMeans(n_clusters = 2, random_state = 0, n_init="auto")
        cm.fit(flat_img)
        labels=cm.labels_
        cl_img = np.reshape(labels, img.shape)
        cl_img = label2rgb(cl_img, bg_label = 0) * np.max(img)
        cl_img[cl_img > 0] = 1
        self.clustered_img = cl_img[:, :, 0]
        
    
    def detect_contours(self):
        contour_img = self.clustered_img.astype(np.uint8)
        f_contours = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = f_contours[0] if len(f_contours) == 2 else f_contours[1]
        
    
    def throw_small_contours(self):
        
        index = 1
        for cntr in self.contours:
            area = cv2.contourArea(cntr)
            x,y,w,h = cv2.boundingRect(cntr)
            if w < 80:
                self.clustered_img[y:y+h, x:x+w, 0] = 0

            index = index + 1

        
