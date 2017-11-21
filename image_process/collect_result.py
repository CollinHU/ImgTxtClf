from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import Image

def load_model(filename, model):
    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    best_prec = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    return model,best_prec

data_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = '/run/shm/common/images/'
use_gpu = torch.cuda.is_available()
print(use_gpu)
######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^

def test_model(data_folder,model):
    directory = data_dir + data_folder
    category_list = os.listdir(directory)

    img_name = []
    img_pred = []

    since = time.time()
    count = 0
    for item in category_list:
        sub_dir = directory + '/' + item
        for img in os.listdir(sub_dir):
            img_dir = sub_dir + '/' + img
            try:
                inputs = Image.open(img_dir)
                inputs = data_transforms(inputs).unsqueeze_(0)
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                else:
                    inputs = Variable(inputs)
                outputs = model(inputs)
                _, preds = torch.topk(outputs.data,5, 1)
                img_name.append(img.split('.')[0])
                img_pred += list(preds.cpu().numpy())#.append(preds.cpu().numpy()[0])
                count += 1
		if count == 10:
			break
            except:
                continue
    
    dict_pred = {'img':img_name, 'pred':img_pred}
    df = pd.DataFrame(data = dict_pred)
    df.to_csv('img_{}_pred.csv'.format(data_folder))
    time_elapsed = time.time() - since
    print('{} complete in {:.0f}m {:.0f}s'.format(data_folder, time_elapsed // 60, time_elapsed % 60))
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#

model_ft = models.resnet34(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 101)

if use_gpu:
    model_ft = model_ft.cuda()
m_acc = 0.0
model_ft, m_acc = load_model('resnet34_checkpoint.pth.tar',model_ft)
print(m_acc)
model_ft.train(False)
fold_list = ['train', 'test', 'val']
for item in fold_list:
    test_model(item, model_ft)


