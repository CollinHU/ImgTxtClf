import get_train_test as dataset
from sklearn.svm import SVC
import pickle
import numpy as np
import nltk

# Train data preprocessing

X_data = dataset.load_dataset()
target_name = X_data['categories']

X_train = X_data['train']
print(X_train["target"][:3])
X_test = X_data['test']

X_train_data = pickle.load(open('../data/train_data_ngram_1.pkl','rb'))
X_test_data = pickle.load(open('../data/test_data_ngram_1.pkl','rb'))

Y_train = X_train['target']
Y_test = X_test['target']

def train_test_model(kernel):
    svm_text_clf = SVC(kernel = kernel)
    svm_text_clf = svm_text_clf.fit(X_train_data, Y_train)

    svm_predicted = svm_text_clf.predict(X_test_data)
    svm_accuracy = np.mean(svm_predicted == Y_test)
    record = open('svm_text_clf_record.txt','a')
    record.write('using svm  with setting: \n')
    record.write('kernel = {} '.format(kernel))
    record.write('the accuracy is: ' + str(svm_accuracy)+'\n')
    record.write('\n')
    record.close()

train_test_model('rbf')
#alpha_list = [1, 0.1, 0.01, 0.001]
#fit_prior_list = [True, False]
'''
for alpha in alpha_list:
    for fit_prior in fit_prior_list:
        train_test_model(alpha, fit_prior)
'''
