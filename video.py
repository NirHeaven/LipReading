from scipy import ndimage
import glob
import os
import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
from statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip
import torch
import numpy as np


class Video(object):
    
    def __init__(self, video_path):
        self._from_dir(video_path)
        
    def _from_dir(self, video_path, img_c=3, img_h=50, img_w=100):
        self.files    = sorted([os.path.join(video_path, x) for x in os.listdir(video_path)])
        self.n_frames = len(self.files)        
        self.speaker  = video_path.split(os.sep)[-2]
        self.name     = os.path.split(video_path)[-1]
        self.img_c    = img_c
        self.img_w    = img_w
        self.img_h    = img_h
    
    # [left, right)
    def get_frames(self, left, right, padding=75):
        filelist = self.files[left:right]        
        frames = [ndimage.imread(im) for im in filelist]
        n = len(frames)
        frames = [np.zeros((self.img_h, self.img_w, self.img_c), dtype=np.uint8)]*(padding-n)+frames
        return self._augmentation(frames)
    
    # flip
    def _augmentation(self, frames):
        n = len(frames)
        temporalvolume = torch.FloatTensor(n, self.img_c, self.img_h, self.img_w)
        # Random Flip
        flip = StatefulRandomHorizontalFlip(0.5)

        for i in range(n):
            temporalvolume[i] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_h, self.img_w)),                
                flip,
                transforms.ToTensor(),
                transforms.Normalize([0, 0, 0], [1, 1, 1])
            ])(frames[i])
            
        return temporalvolume.permute(1, 0, 2, 3).contiguous()
        