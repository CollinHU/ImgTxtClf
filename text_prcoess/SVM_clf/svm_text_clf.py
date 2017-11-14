import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import get_train_test as dataset
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import nltk
import gc
# Train data preprocessing

X_data = dataset.load_dataset()
target_name = X_data['categories']

X_train = X_data['train']
print(X_train["target"][:3])
X_test = X_data['test']

print('reading data')
X_train_data = pd.read_csv('/mnt/zyhu/data/train_tfidf_w2v_df.csv',index_col = 0)
X_train_data = X_train_data.values
#X_test_data = pickle.load(open('../data/test_data_ngram_1.pkl','rb'))
X_test_data = pd.read_csv('/mnt/zyhu/data/test_tfidf_w2v_df.csv',index_col = 0)
X_test_data = X_test_data.values

Y_train = X_train['target']
Y_test = X_test['target']

def train_test_model(kernel):
    svm_text_clf = SVC(kernel = kernel)
    print("training")
    svm_text_clf = svm_text_clf.fit(X_train_data, Y_train)

    print("do predcition")
    svm_predicted = svm_text_clf.predict(X_train_data)
    svm_accuracy = np.mean(svm_predicted == Y_train)
    
    record = open('testing_speed.txt','a')
    record.write('using svm  with setting: \n')
    record.write('kernel = {} '.format(kernel))
    record.write('training data the accuracy is: ' + str(svm_accuracy)+'\n')

    svm_predicted = svm_text_clf.predict(X_test_data)
    svm_accuracy = np.mean(svm_predicted == Y_test)
    record.write('test data the accuracy is: ' + str(svm_accuracy)+'\n')
    record.write('\n')
    record.close()

print('train and test model')
kernel_list = ['rbf', 'linear','sigmoid']#,'poly']
for item in kernel_list:
    print(item)
    train_test_model(item)
    gc.collect()

#alpha_list = [1, 0.1, 0.01, 0.001]
#fit_prior_list = [True, False]
print('finish')
'''
for alpha in alpha_list:
    for fit_prior in fit_prior_list:
        train_test_model(alpha, fit_prior)
'''
