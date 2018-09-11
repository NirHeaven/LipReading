from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os
import time
from common import *
from LipNet import LipNet
import time
import torch.nn as nn
import torch
import torch.optim as optim
import editdistance


def array2sentence(array):
    words = []
    s = ''
    last = 0
    for i in range(array.shape[0]):
        p = np.argmax(array[i,:])
        if(p != last and p != 0):
            s += int2char(p)
        last = p
    return s

def validation(model, weights, dataloader):
    if(weights):
        model.load_state_dict(torch.load(weights))    
    model.eval()
    result = []
    for i, batch in enumerate(dataloader):
        video_input     = batch['video_input'].cuda()
        video_lengths   = batch['video_length'].long()       
        targets_input   = batch['targets_input'].long()
        targets_lengths = torch.from_numpy(batch['targets_length'].numpy()).long()
        acts = model(video_input)
        
        output = acts.detach().cpu().numpy()
        for i in range(output.shape[0]):
            current = output[i].reshape(output.shape[1], output.shape[2])
            ground_truch_length = targets_lengths[i]
            ground_truch = vector_to_word(list(targets_input[i].numpy())[0:ground_truch_length])          

            prediction = array2sentence(current)
            result.append([ground_truch, prediction])
    return result
            
def cer(result):
    cer = [editdistance.eval(p[0], p[1])/len(p[0]) for p in result]
    return np.array(cer).mean()
    
def wer(result):
    word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in result]
    wer = [editdistance.eval(p[0], p[1])/len(p[0]) for p in word_pairs]
    return np.array(wer).mean()