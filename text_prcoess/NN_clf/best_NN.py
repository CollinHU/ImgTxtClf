#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:55:59 2017

@author: tianxiangzhang
"""

import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import gc
import copy
from sklearn.preprocessing import StandardScaler
# Train data preprocessing

def save_checkpoint(state, filename='nn_checkpoint.pth.tar'):
    torch.save(state, filename)

standar_tran = StandardScaler()
# Train data preprocessing

print('reading data')
X_train = pd.read_csv('/Users/collin/myDocuments/BDT5001_data/common/texts/train_tfidf_w2v_df.csv',index_col = 0)
X_train_data = X_train.iloc[:,:200].values
#X_test_data = pickle.load(open('../data/test_data_ngram_1.pkl','rb'))

X_val = pd.read_csv('/Users/collin/myDocuments/BDT5001_data/common/texts/val_tfidf_w2v_df.csv',index_col = 0)
X_val_data = X_val.iloc[:,:200].values

#X_train_data = np.append(X_train_data, X_val_data, axis = 0)

X_test = pd.read_csv('/Users/collin/myDocuments/BDT5001_data/common/texts/test_tfidf_w2v_df.csv',index_col = 0)
X_test_data = X_test.iloc[:,:200].values

Y_train = X_train['category_id'].values
Y_val = X_val['category_id'].values
#Y_train = np.append(Y_train,Y_val, axis = 0)

Y_test = X_test['category_id'].values

#X_train_data = X_train_data
print(X_train_data.shape)
#Y_train = Y_train

#X_test_data = X_test_data
#Y_test = Y_test
X_tr_tst = np.append(X_train_data,X_test_data,axis = 0)
X_tr_tst = np.append(X_tr_tst, X_val_data, axis = 0)
standar_tran = standar_tran.fit(X_tr_tst)

X_train_data = standar_tran.transform(X_train_data)
X_val_data = standar_tran.transform(X_val_data)
X_test_data = standar_tran.transform(X_test_data)

train_size = len(Y_train)
test_size = len(Y_test)
val_size = len(Y_val)
print(train_size)
print(test_size)

Y_train_v = np.zeros((train_size,101))
Y_val_v = np.zeros((val_size, 101))
Y_test_v = np.zeros((test_size,101))

Y_train_v[np.arange(train_size), Y_train] = 1
Y_val_v[np.arange(val_size), Y_val] = 1
Y_test_v[np.arange(test_size), Y_test] = 1
#Y_train = Y_train_v#.astype(int)
#Y_test = Y_test_v#.astype(int)
#print Y_train.shape


#processing input data and label

X_train_data = Variable(torch.from_numpy(X_train_data).float())
X_val_data = Variable(torch.from_numpy(X_val_data).float())
X_test_data = Variable(torch.from_numpy(X_test_data).float())
Y_train_v = Variable(torch.from_numpy(Y_train_v).float(), requires_grad=False)
Y_val_v = Variable(torch.from_numpy(Y_val_v).float(), requires_grad=False)
Y_test_v = Variable(torch.from_numpy(Y_test_v).float(),requires_grad=False)

#print Y_train[0,:]
#print Y_train.shape
#print Y_train[0,:]
model_file = 'sd_nn_checkpoint.pth.tar'
batch_size,D_in,H,D_out = 1000,200,500,101
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax()
)

loss_fn =  torch.nn.MSELoss(size_average=False)
# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Variables it should update.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_acc = 0.0
itr_num = int(train_size / batch_size) + 1
for t in range(1000):
    # Forward pass: compute predicted y by passing x to the model.
    total_loss = 0
    acc = 0
    for i in range(itr_num):
            
        start_p = i * batch_size
        if start_p >= train_size:
            break;
        end_p = (i + 1) * batch_size
        if end_p > train_size:
            end_p = train_size
 #           print(end_p)
        
        x = X_train_data[start_p:end_p, :]
        y = Y_train_v[start_p:end_p, :]
        y_target = Y_train[start_p:end_p].reshape(-1,1)
        
        y_pred = model(x)

    # Compute and print loss.
    #    print(y_pred.data[0])
        loss = loss_fn(y_pred, y)
        total_loss += loss.data[0]
        _,index = torch.max(y_pred,1)
        #print(index.data.numpy().shape)
        acc += np.sum(index.data.numpy() == y_target)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
        optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
        loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
        optimizer.step()
    print('train',t, total_loss/train_size)
    print(float(acc)/train_size)
    if t == 100:
        learning_rate /= 2
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if t == 150:
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if t >= 250 and t % 250 == 0:
        learning_rate /= 2
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    val_itr_num = int(val_size/batch_size)+1
    val_acc = 0.0
    val_loss = 0.0
    for i in range(val_itr_num):
            
        start_p = i * batch_size
        if start_p >= val_size:
            break;
        end_p = (i + 1) * batch_size
        if end_p > val_size:
            end_p = val_size
#            print(end_p)
        
        x = X_val_data[start_p:end_p, :]
        y = Y_val_v[start_p:end_p, :]
        y_target = Y_val[start_p:end_p].reshape(-1,1)
        
        y_pred = model(x)

    # Compute and print loss.
    #    print(y_pred.data[0])
        loss = loss_fn(y_pred, y)
        val_loss += loss.data[0]
        _,index = torch.max(y_pred,1)
        #print(index.data.numpy().shape)
        val_acc += np.sum(index.data.numpy() == y_target)
    print('val', val_loss/val_size)
    print(float(val_acc)/val_size)
    if best_acc < val_acc:
        best_acc = val_acc/val_size
        best_model = copy.deepcopy(model)
        save_checkpoint({'epoch': t+1,
        'state_dict': best_model.state_dict(),
        'best_prec1': best_acc,
        'optimizer' : optimizer.state_dict(),},model_file)


itr_num = int(test_size / batch_size) + 1
    # Forward pass: compute predicted y by passing x to the model.
total_loss = 0
acc = 0
checkpoint = torch.load(model_file)
model.load_state_dict(checkpoint['state_dict'])
for i in range(itr_num):
    start_p = i * batch_size
    if start_p >= test_size:
        break;
    end_p = (i + 1) * batch_size
    if end_p > test_size:
        end_p = test_size
        
    x = X_test_data[start_p:end_p, :]
    y = Y_test_v[start_p:end_p, :]
    y_target = Y_test[start_p:end_p].reshape(-1,1)
        
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    total_loss += loss.data[0]
    _,index = torch.max(y_pred,1)
    #print(index.data.numpy().shape)
    acc += np.sum(index.data.numpy() == y_target)

print('test',total_loss/test_size)
print(float(acc)/test_size)
