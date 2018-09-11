from dataloader import GRIDDataset
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
import random
from val import validation, cer, wer
import csv
from tensorboardX import SummaryWriter
import math
import toml
import sys
import logging
import shutil

logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s:: ", level=logging.INFO, stream=sys.stdout)

# Load options
with open('options.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())
print(options)

# Tensorboard Summary Writer
writer = SummaryWriter()

# Initialize random seed
random.seed(options['general']['python_seed']) 
torch.manual_seed(options['general']['torch_seed'])
torch.cuda.manual_seed_all(options['general']['torch_cuda_seed'])

# cuda
os.environ['CUDA_VISIBLE_DEVICES'] = options['general']['gpus']


# train loader
train        = GRIDDataset(options['train']['path'], options['train']['align'], 75)
train_dl     = DataLoader(train, batch_size=options['train']['batch_size'],
                        shuffle=True, num_workers=options['train']['num_threads'])
                        
# validation loader
if(options['validation']['validate']):
    val          = GRIDDataset(options['validation']['path'], options['validation']['align'], 75)
    val_dl       = DataLoader(val, batch_size=options['validation']['batch_size'],
                        shuffle=True, num_workers=options['validation']['num_threads'])
                        
# model
model     = LipNet(options['params'])
if(options['model']['load_weights']):
    if not torch.cuda.is_available():
        state_dict = torch.load(options['model']['pretrained'], map_location='cpu')

    else:
        state_dict = torch.load(options['model']['pretrained'])
    if 'wer' in state_dict:
        del state_dict['wer']
    if 'cer' in state_dict:
        del state_dict['cer']
    model.load_state_dict(state_dict)
    

net       = torch.nn.DataParallel(model).cuda()
ctc_loss  = nn.CTCLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# iteration
n_iter = 0
if(not os.path.exists(options['snapshot']['save_dir'])):
    os.mkdir(options['snapshot']['save_dir'])
save_prefix = os.path.join(options['snapshot']['save_dir'], options['snapshot']['save_prefix'])


def batch_optim(batch, net, loss, optim, ratio, epoch):
    optim.zero_grad()
    video_input     = batch['video_input']
    video_lengths   = batch['video_length'].int().reshape(-1)
    targets_input   = batch['targets_input'].int()
    targets_lengths = batch['targets_length'].int().reshape(-1)
    if torch.cuda.is_available():
        video_input = video_input.cuda()
        video_lengths = video_lengths.cuda()
        targets_input = targets_input.cuda()
        targets_lengths = targets_lengths.cuda()
    acts = net(video_input, video_lengths)
    acts = acts.permute(1, 0, 2)
    loss = loss(acts, targets_input, video_lengths, targets_lengths)
    loss.backward()
    if(options['train']['train']):
        for w in model.parameters():
            w.grad *= math.pow(ratio, epoch)
        optimizer.step()
    return loss.cpu().detach().numpy()*video_input.size(2)
best_c = 1.
best_w = 1.
for epoch in range(options['train']['tot_epoch']):
    
    for i, batch in enumerate(train_dl):
        model.train()
        start = time.time()
        
        loss = batch_optim(batch, net, ctc_loss, optimizer, 1, epoch)

        if(n_iter % options['train']['summary_iter'] == 0):
            logging.info('epoch:{}, niter:{}, loss: {}, s/per second: {}'.
                         format(epoch, n_iter, loss, (time.time() - start) / options['train']['batch_size']))
            writer.add_scalar('loss', loss, n_iter)


        if(n_iter != 0 and n_iter % options['snapshot']['save_iter'] == 0):
            state_dict = model.state_dict()

            result = validation(model, None, val_dl)
            c, w = cer(result), wer(result)
            logging.info('epoch: {}, niter: {}, cer: {}, wer: {}'
                         .format(epoch, n_iter, c, w))
            state_dict['cer'] = c
            state_dict['wer'] = w
            torch.save(state_dict, save_prefix + '-{}-{}.pkl'.format(epoch, n_iter))
            if c < best_c:
                best_c = c
                shutil.copy(save_prefix + '-{}-{}.pkl'.format(epoch, n_iter),
                            save_prefix + '-{}-{}-best_cer.pkl'.format(epoch, n_iter))
            if w < best_w:
                best_w = w
                shutil.copy(save_prefix + '-{}-{}.pkl'.format(epoch, n_iter),
                            save_prefix + '-{}-{}-best_wer.pkl'.format(epoch, n_iter))
            # validation outputs

            csv_datas = [['ground truth', 'prediction']] + result
            with open(save_prefix + '-result-{}-{}.csv'.format(epoch, n_iter), 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerows(csv_datas)

            # training process
            rows = []
            if (os.path.exists(save_prefix + '-process.csv')):
                with open(save_prefix + '-process.csv', 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        rows.append(row)
            else:
                rows.append(['iter', 'cer', 'wer'])

            rows.append([n_iter, c, w])
            with open(save_prefix + '-process.csv', 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerows(rows)


    n_iter += 1

writer.close()
    