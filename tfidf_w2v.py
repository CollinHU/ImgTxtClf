import gensim
import get_train_test as dataset
import numpy as np 
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.snowball import SnowballStemmer
import pickle
from collections import defaultdict
# let X be a list of tokenized texts (i.e. list of lists of tokens)

stemmer = SnowballStemmer("english", ignore_stopwords = True)
class StemmedTfidfCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfCountVectorizer, self).build_analyzer()
        return lambda doc:([stemmer.stem(w) for w in analyzer(doc)])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y = None):
        tfidf = StemmedTfidfCountVectorizer(stop_words = 'english', min_df = 11)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        X = [word_tokenize(s) for s in X]
        X = [[stemmer.stem(w) for w in s] for s in X]
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


X_data = dataset.load_dataset()
X_train = X_data['train']
X_test = X_data['test']

data = X_train['data'] + X_test['data']

print('processing w2v data')
w2v_data = [word_tokenize(s) for s in data]
stemmed_w2v_data = [[stemmer.stem(w) for w in s] for s in w2v_data]

print('train w2v model')
model = gensim.models.Word2Vec(stemmed_w2v_data, size=200)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))

print('train w2v-tfidf')
tfidf_w2v = TfidfEmbeddingVectorizer(w2v)
tfidf_w2v.fit(data)

train_tfidf_w2v = tfidf_w2v.transform(X_train['data'])
#dic_train = {'w2v':train_tfidf_w2v}
train_tfidf_w2v_df = pd.DataFrame(data = train_tfidf_w2v)
train_tfidf_w2v_df.to_csv('train_tfidf_w2v_df.csv')

test_tfidf_w2v = tfidf_w2v.transform(X_test['data'])
#dic_test = {'w2v':test_tfidf_w2v}
test_tfidf_w2v_df = pd.DataFrame(data = test_tfidf_w2v)
test_tfidf_w2v_df.to_csv('test_tfidf_w2v_df.csv')
'''
train_data_converted = txt2wordvector(X_train['data'],'train')
#store the content
with open("train_data_ngram_2.pkl", 'wb') as handle:
                        pickle.dump(train_data_converted, handle)
print('processing test data')
test_data_converted = txt2wordvector(X_test['data'],'test')
#store the content
with open("test_data_ngram_2.pkl", 'wb') as t_handle:
                        pickle.dump(test_data_converted, t_handle)
'''
