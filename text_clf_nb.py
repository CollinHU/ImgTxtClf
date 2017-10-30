#from sklearn.datasets import fetch_20newsgroups
import get_train_test as dataset
import numpy as np 

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline 

import nltk
from nltk.stem.snowball import SnowballStemmer

# Train data preprocessing

X_data = dataset.load_dataset()
target_name = X_data['categories']

X_train = X_data['train']
#print("testing\n")
print(target_name.keys())
#print('\n'.join(X_train.data[0].split('\n')[:3]))
print(X_train["target"][:3])
#Test data preprocessing
X_test = X_data['test']

stemmer = SnowballStemmer("english", ignore_stopwords = True)

class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc:([stemmer.stem(w) for w in analyzer(doc)])

NB_text_clf = Pipeline([('vect',StemmedCountVectorizer(stop_words = 'english')),
						('tfidf', TfidfTransformer()),
						('mnb',MultinomialNB())])

stemmed_parameter = {'vect__ngram_range':[(1,1),(1,2)],
					'tfidf__use_idf':[True,False],
					'mnb__alpha':[1e-2,1e-3],}

model_selection_stemmed = GridSearchCV(NB_text_clf, stemmed_parameter, n_jobs = -1)
model_selection_stemmed = model_selection_stemmed.fit(X_train['data'], X_train['target'])
#NB_text_clf = NB_text_clf.fit(X_train['data'],X_train['target'])
#NB_predicted = NB_text_clf.predict(X_test['data'])
#NB_accuracy = np.mean(NB_predicted == X_test['target'])
#print("the accuracy of Naive Bayes is {}".format(NB_accuracy))
print(model_selection_stemmed.best_score_)
print(model_selection_stemmed.best_params_)

