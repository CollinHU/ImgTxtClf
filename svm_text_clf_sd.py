import get_train_test as dataset
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import nltk
import gc

from sklearn.preprocessing import StandardScaler
# Train data preprocessing

standar_tran = StandardScaler()

X_data = dataset.load_dataset()
target_name = X_data['categories']

X_train = X_data['train']
print(X_train["target"][:3])
X_test = X_data['test']

print('reading data')
X_train_data = pd.read_csv('../data/train_tfidf_w2v_df.csv',index_col = 0)
X_train_data = X_train_data.values
X_train_data = standar_tran.fit_transform(X_train_data)

#X_test_data = pickle.load(open('../data/test_data_ngram_1.pkl','rb'))
X_test_data = pd.read_csv('../data/test_tfidf_w2v_df.csv',index_col = 0)
X_test_data = X_test_data.values
X_test_data = standar_tran.fit_transform(X_test_data)

Y_train = X_train['target']
Y_test = X_test['target']

def train_test_model(kernel):
    svm_text_clf = SVC(kernel = kernel)
    svm_text_clf = svm_text_clf.fit(X_train_data, Y_train)

    svm_predicted = svm_text_clf.predict(X_train_data)
    svm_accuracy = np.mean(svm_predicted == Y_train)
    
    record = open('svm_text_clf_record_sd.txt','a')
    record.write('using svm  with setting: \n')
    record.write('kernel = {} '.format(kernel))
    record.write('training data the accuracy is: ' + str(svm_accuracy)+'\n')

    svm_predicted = svm_text_clf.predict(X_test_data)
    svm_accuracy = np.mean(svm_predicted == Y_test)
    record.write('test data the accuracy is: ' + str(svm_accuracy)+'\n')
    record.write('\n')
    record.close()

print('train and test model')
train_test_model('linear')
'''
kernel_list = ['rbf', 'linear','sigmoid','poly']
for item in kernel_list:
    train_test_model(item)
    gc.collect()
'''
#alpha_list = [1, 0.1, 0.01, 0.001]
#fit_prior_list = [True, False]
print('finish')
'''
for alpha in alpha_list:
    for fit_prior in fit_prior_list:
        train_test_model(alpha, fit_prior)
'''
