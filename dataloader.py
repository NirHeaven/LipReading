import os
import glob
import numpy as np
import pickle as pkl
from common import *
from torch.utils.data import Dataset, DataLoader
import random

from align import Align
from video import Video


class GRIDDataset(Dataset):
    
    def __init__(self, 
        video_path, 
        align_path, 
        video_padding_length,
    ):
        self.video_path = video_path
        self.align_path = align_path
        self.video_padding_length = video_padding_length        
        
        self.videos     = glob.glob(os.path.join(video_path, "*", "*"))
        self.n_videos   = len(self.videos)
        
    def __len__(self):
        return self.n_videos
    
    
    def __getitem__(self, index):
        file = self.videos[index]
        
        video = Video(file)
        align = Align(os.path.join(self.align_path, video.speaker, 'align', video.name+".align"))
        
        
        return {'video_input': video.get_frames(0, video.n_frames, self.video_padding_length),
                'video_length': self.video_padding_length,
                'targets_input': align.sentence(),
                'targets_length': align.sentence_length,
               }