import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py
import get_train_test as dataset
from sklearn.naive_bayes import MultinomialNB
import pickle
import numpy as np
import nltk

# Train data preprocessing

X_data = dataset.load_dataset()
target_name = X_data['categories']

X_train = X_data['train']
print(X_train["target"][:3])
X_test = X_data['test']

X_train_data = pickle.load(open('train_data_ngram_1.pkl','rb'))
X_test_data = pickle.load(open('test_data_ngram_1.pkl','rb'))

Y_train = X_train['target']
Y_test = X_test['target']

def train_test_model(alpha, fit_prior):
    NB_text_clf = MultinomialNB(alpha = alpha,fit_prior = fit_prior)
    NB_text_clf = NB_text_clf.fit(X_train_data, Y_train)

    NB_predicted = NB_text_clf.predict(X_test_data)
    NB_accuracy = np.mean(NB_predicted == Y_test)
    record = open('nb_text_clf_record.txt','a')
    record.write('using multinomial NB with setting: \n')
    record.write('ngram = 1')
    record.write('alpha = {} '.format(alpha))
    record.write('fit_prior = {} \n'.format(fit_prior))
    record.write('the accuracy is: ' + str(NB_accuracy)+'\n')
    record.write('\n')
    record.close()

alpha_list = [1, 0.1, 0.01, 0.001]
fit_prior_list = [True, False]

for alpha in alpha_list:
    for fit_prior in fit_prior_list:
        train_test_model(alpha, fit_prior)

