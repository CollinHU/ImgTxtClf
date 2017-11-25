from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import os


def load_model(filename, model):
    checkpoint = torch.load(filename)#, map_location=lambda storage, loc: storage)
    epoch = checkpoint['epoch']
    best_prec = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    return model, best_prec

data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/run/shm/common/images'
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val','test']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=64,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val','test']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val','test']}
dset_classes = dsets['train'].classes
print(dset_classes)
category = {}
for i  in range(len(dset_classes)):
    category[dset_classes[i]] = i
import json

with open('category_id.txt', 'w') as f:
        f.write(json.dumps(category))
use_gpu = torch.cuda.is_available()
print(use_gpu)
######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def save_checkpoint(state, filename='resnet18_checkpoint.pth.tar'):
    torch.save(state, filename)

def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dset_loaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dset_classes[labels.data[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

######################################################################

def test_model(model, criterion, phase):
    since = time.time()
        # Each epoch has a training and validation phase
    model.train(False)  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    count = 0
            # Iterate over data.
    for data in dset_loaders[phase]:
        inputs, labels = data
        count += 1
                # wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), \
                    Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.topk(outputs.data, 5, 1)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        running_loss += loss.data[0]
        for i in range(5):
            running_corrects += torch.sum(preds[:, i] == labels.data)
        if count % 100 == 0:
            print(count)
        
    epoch_loss = running_loss / dset_sizes[phase]
    epoch_acc = running_corrects / dset_sizes[phase]
    print(dset_sizes[phase])
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


    time_elapsed = time.time() - since
    print('{} complete in {:.0f}m {:.0f}s'.format(phase, time_elapsed // 60, time_elapsed % 60))
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#

model_ft = models.resnet152(pretrained=False)
for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 101)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
if use_gpu:
    model_ft = model_ft.cuda()
m_acc = 0.0

model_ft,m_acc = load_model('resnet152_checkpoint.pth.tar',model_ft)
print(m_acc)
dir_list = ['test', 'val', 'train']
for item in dir_list:
	test_model(model_ft, criterion, item)
