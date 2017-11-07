import get_train_test as dataset
import numpy as np 
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.snowball import SnowballStemmer
import pickle
from nltk.corpus import stopwords
#print("loading data")


X_data = dataset.load_dataset()

X_train = X_data['train']
X_test = X_data['test']

stemmer = SnowballStemmer("english", ignore_stopwords = True)

class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc:([stemmer.stem(w) for w in analyzer(doc)])


StemmedConvert = StemmedCountVectorizer(stop_words = 'english',min_df = 11)
TfidfTran = TfidfTransformer()
def txt2wordvector(data,mod): 
    if mod == 'train':
        data = StemmedConvert.fit_transform(data)
        data = TfidfTran.fit_transform(data)
    elif mod == 'test':
        data = StemmedConvert.transform(data)
        data = TfidfTran.transform(data)
    else:
        print("please setting convert model")
        return None
    print(data.shape)
    print(type(data))
    return data

t_data = X_train['data']+X_test['data']
train_size = len(X_train['data'])
data_size = len(t_data)
print('processing train data')
data_converted = txt2wordvector(t_data,'train')
#store the content
train_data_converted = data_converted[0:train_size]
with open("../data/train_data_ngram_1.pkl", 'wb') as handle:
                        pickle.dump(train_data_converted, handle)


print('processing test data')
test_data_converted = data_converted[train_size:data_size]
#store the content
with open("../data/test_data_ngram_1.pkl", 'wb') as t_handle:
                        pickle.dump(test_data_converted, t_handle)

