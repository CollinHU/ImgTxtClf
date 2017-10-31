import get_train_test as dataset
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.snowball import SnowballStemmer
import pickle
# Train data preprocessing
#print("loading data")
X_data = dataset.load_dataset()

X_train = X_data['train']
X_test = X_data['test']

stemmer = SnowballStemmer("english", ignore_stopwords = True)
class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc:([stemmer.stem(w) for w in analyzer(doc)])

def txt2wordvector(data):
    StemmedConvert = StemmedCountVectorizer(stop_words = 'english')
    TfidfTran = TfidfTransformer()
    data = StemmedConvert.fit_transform(data)
    data = TfidfTran.fit_transform(data)
    print(data.shape)
    print(type(data))
    return data

print('processing train data')
train_data_converted = txt2wordvector(X_train['data'])
#store the content
with open("train_data.pkl", 'wb') as handle:
                        pickle.dump(train_data_converted, handle)


print('processing test data')
test_data_converted = txt2wordvector(X_test['data'])
#store the content
with open("test_data.pkl", 'wb') as t_handle:
                        pickle.dump(test_data_converted, t_handle)


