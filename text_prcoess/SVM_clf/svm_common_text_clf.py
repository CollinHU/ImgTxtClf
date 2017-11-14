from sklearn.svm import SVC
import pandas as pd
import numpy as np
import nltk
import gc
# Train data preprocessing

print('reading data')
X_train = pd.read_csv('/mnt/zyhu/common/texts/train_tfidf_w2v_df.csv',index_col = 0)
X_train_data = X_train.iloc[:,:200].values
#X_test_data = pickle.load(open('../data/test_data_ngram_1.pkl','rb'))

X_val = pd.read_csv('/mnt/zyhu/common/texts/val_tfidf_w2v_df.csv',index_col = 0)
X_val_data = X_val.iloc[:,:200].values

X_train_data = np.append(X_train_data, X_val_data, axis = 0)

X_test = pd.read_csv('/mnt/zyhu/common/texts/test_tfidf_w2v_df.csv',index_col = 0)
X_test_data = X_test.iloc[:,:200].values

Y_train = X_train['category_id'].values
Y_val = X_val['category_id'].values
Y_train = np.append(Y_train,Y_val, axis = 0)

Y_test = X_test['category_id'].values

def train_test_model(kernel):
    svm_text_clf = SVC(kernel = kernel)
    svm_text_clf = svm_text_clf.fit(X_train_data, Y_train)

    svm_predicted = svm_text_clf.predict(X_train_data)
    svm_accuracy = np.mean(svm_predicted == Y_train)
    
    record = open('svm_common_text_clf_record_02.txt','a')
    record.write('using svm  with setting: \n')
    record.write('kernel = {} '.format(kernel))
    record.write('training data the accuracy is: ' + str(svm_accuracy)+'\n')

    svm_predicted = svm_text_clf.predict(X_test_data)
    svm_accuracy = np.mean(svm_predicted == Y_test)
    record.write('test data the accuracy is: ' + str(svm_accuracy)+'\n')
    record.write('\n')
    record.close()

print('train and test model')
kernel_list = ['sigmoid','rbf','linear'] #,'poly']
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
