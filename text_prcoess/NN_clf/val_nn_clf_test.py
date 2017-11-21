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

#standar_tran = StandardScaler()
# Train data preprocessing
standar_tran = StandardScaler()

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

X_tr_tst = np.append(X_train_data,X_test_data,axis = 0)
X_tr_tst = np.append(X_tr_tst, X_val_data, axis = 0)
standar_tran = standar_tran.fit(X_tr_tst)

X_train_data = standar_tran.transform(X_train_data)
X_val_data = standar_tran.transform(X_val_data)
X_test_data = standar_tran.transform(X_test_data)

X_train_data = Variable(torch.from_numpy(X_train_data).float())
X_val_data = Variable(torch.from_numpy(X_val_data).float())
X_test_data = Variable(torch.from_numpy(X_test_data).float())

Y_train_v = Variable(torch.from_numpy(Y_train_v).float(), requires_grad=False)
Y_val_v = Variable(torch.from_numpy(Y_val_v).float(), requires_grad=False)
Y_test_v = Variable(torch.from_numpy(Y_test_v).float(),requires_grad=False)

batch_size,D_in,H,D_out = 1000,200,500,101
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax()
)

loss_fn =  torch.nn.MSELoss(size_average=False)

def get_prediction(model, X_data, labels, labels_v, data_size, df):
    itr_num = int(data_size / batch_size) + 1
    print(data_size)
    # Forward pass: compute predicted y by passing x to the model.
    total_loss = 0
    acc = 0

    for i in range(itr_num):
        start_p = i * batch_size
        if start_p >= data_size:
            break;
        end_p = (i + 1) * batch_size
        if end_p > data_size:
            end_p = data_size
            #print(end_p)
            
        x = X_data[start_p:end_p, :]
        #print(x.data.numpy().shape)
        y = labels_v[start_p:end_p, :]
        y_target = labels[start_p:end_p].reshape(-1,1)
            
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        total_loss += loss.data[0]
        _,index = torch.topk(y_pred,5,1)
        if i == 0:
            pred_result = list(index.data.numpy())
            #print(pred_result)
        else:
            pred_result += list(index.data.numpy())
        #print(index.data.numpy().shape)
        for i in range(5):
            #print(index.data.numpy()[:,i].shape)
            acc += np.sum(index.data.numpy()[:,i].reshape(-1,1) == y_target)

    print('test',total_loss/data_size)
    print(float(acc)/data_size)
    df['pred_label'] = pred_result
    df = df[['text_name','category_id','pred_label']]
    df.to_csv('tmp.csv')



checkpoint = torch.load('sd_nn_checkpoint.pth.tar')
epoch_acc = checkpoint['best_prec1']

print(epoch_acc)
model.load_state_dict(checkpoint['state_dict'])

get_prediction(model, X_val_data, Y_val, Y_val_v, val_size, X_val)
