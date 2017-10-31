import get_train_test as dataset
import numpy as np 

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline 
from sklearn.svm import SVC
import nltk
from nltk.stem.snowball import SnowballStemmer

# Train data preprocessing

X_data = dataset.load_dataset()
target_name = X_data['categories']

X_train = X_data['train']
#print("testing\n")
#print(target_name.keys())
#print('\n'.join(X_train.data[0].split('\n')[:3]))
print(X_train["target"][:3])
#Test data preprocessing
X_test = X_data['test']

stemmer = SnowballStemmer("english", ignore_stopwords = True)

class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc:([stemmer.stem(w) for w in analyzer(doc)])

#for alpha = 1, ngram = (1,1) use_idf = true accuaracy is 0.69
'''
NB_text_clf = Pipeline([('vect',CountVectorizer(stop_words = 'english')),
						('tfidf', TfidfTransformer()),
						('mnb',MultinomialNB(alpha = 1e-3))])

NB_text_clf = NB_text_clf.fit(X_train['data'],X_train['target'])
NB_predicted = NB_text_clf.predict(X_test['data'])
NB_accuracy = np.mean(NB_predicted == X_test['target'])
record = open('nb_clf_accuracy.txt','a')
record.write('using CountVectorizr with stop word mnb with alpha = 1e-3 \n')
record.write('the accuracy is: ' + str(NB_accuracy)+'\n')
record.close()

NB_text_clf = Pipeline([('vect',StemmedCountVectorizer(stop_words = 'english')),
						('tfidf', TfidfTransformer()),
						('mnb',MultinomialNB())])

stemmed_parameter = {'vect__ngram_range':[(1,1),(1,2)],
					'tfidf__use_idf':[True,False],
					'mnb__alpha':[1e-2,1e-3],}

model_selection_stemmed = GridSearchCV(NB_text_clf, stemmed_parameter, n_jobs = -1)
model_selection_stemmed = model_selection_stemmed.fit(X_train['data'], X_train['target'])

record = open('nb_clf_accuracy.txt','a')
record.write('this is mnb experiments\n')
record.write("the socre is: ")
record.write(str(model_selection_stemmed.best_score_)+'\n')
record.write('parameters is ' + str(model_selection_stemmed.best_params_)+'\n')
record.close()
'''
#print("the accuracy of Naive Bayes is {}".format(NB_accuracy))
#print(model_selection_stemmed.best_score_)
#print(model_selection_stemmed.best_params_)
svm_text_clf = Pipeline([('vect',CountVectorizer(stop_words = 'english')),
						('tfidf', TfidfTransformer()),
						('svm',SVC())])

svm_text_clf = svm_text_clf.fit(X_train['data'],X_train['target'])
svm_predicted = svm_text_clf.predict(X_test['data'])
svm_accuracy = np.mean(svm_predicted == X_test['target'])
record = open('nb_clf_accuracy.txt','a')
record,write('using CountVectorizr with stop word svm classifer(default)\n')
record.write('the accuracy is: ' + str(svm_accuracy)+'\n')
record.close()
'''
stemmed_parameter = {'vect__ngram_range':[(1,1),(1,2)],
					'tfidf__use_idf':[True,False],
					'svm__kernel':['linear', 'poly', 'rbf', 'sigmoid']}

model_selection_stemmed = GridSearchCV(svm_text_clf, stemmed_parameter, n_jobs = -1)
model_selection_stemmed = model_selection_stemmed.fit(X_train['data'], X_train['target'])

record = open('nb_clf_accuracy.txt','a')
record.write('this is svm experiments \n')
record.write("the socre is: ")
record.write(str(model_selection_stemmed.best_score_)+'\n')
record.write('parameters is ' + str(model_selection_stemmed.best_params_)+'\n')
record.close()
'''
