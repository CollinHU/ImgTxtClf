import gensim
from gensim.models.keyedvectors import KeyedVectors

import get_train_test as dataset

import numpy as np 

import pandas as pd

from nltk import word_tokenize
import nltk
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from collections import defaultdict
import gc
# let X be a list of tokenized texts (i.e. list of lists of tokens)

stemmer = SnowballStemmer("english", ignore_stopwords = True)

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec,tfidf_dict,tfidf_idf):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())
        self.tfidf_dict = tfidf_dict
        self.tfidf_idf = tfidf_idf

    def fit(self):
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(self.tfidf_idf)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, self.tfidf_idf[i]) for w, i in self.tfidf_dict.items()])

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


def construct_model():
    print('load w2v model')
    word_vectors = KeyedVectors.load_word2vec_format('../data/w2v_dic.txt', binary=False)
    #print(type(word_vectors))
    w2v = dict(zip(word_vectors.index2word, word_vectors.syn0))
    
    print('load tfidf')
    with open('../../data/tfidf_idf_list.txt','r') as f:
        lines = f.readlines()
        tfidf_idf = [float(item.split('\n')[0]) for item in lines]
        
    with open('../../data/tfidf_dic.txt', 'r') as f:
        tfidf_dic  = eval(f.read())
        
    print('construct w2v model')
    tfidf_w2v = TfidfEmbeddingVectorizer(w2v,tfidf_dic,tfidf_idf)
    tfidf_w2v.fit()
    
    gc.collect()
    
    return tfidf_w2v

def data_w2v(tfidf_w2v, data):
    data_size = len(data)
    step = 100
    N = int(data_size/step)
    if N > 1:
        result_w2v = tfidf_w2v.transform(data[0:step])
        for i in range(1, N):
            start_p = i*step
            end_p = (i+1)*step
            print(start_p)
            ap_w2v = tfidf_w2v.transform(data[start_p:end_p])
            result_w2v = np.append(result_w2v,ap_w2v,axis = 0)
            gc.collect()
        if end_p < data_size:
            ap_w2v = tfidf_w2v.transform(data[end_p:data_size])
            result_w2v = np.append(result_w2v,ap_w2v,axis = 0)
    else:
        result_w2v = tfidf_w2v.transform(data)
    
    return result_w2v

X_data = dataset.load_dataset()
X_train = X_data['train']
X_test = X_data['test']

tfidf_w2v = construct_model()

print('processing train')
train_tfidf_w2v = data_w2v(tfidf_w2v,X_train['data'])
train_tfidf_w2v_df = pd.DataFrame(data = train_tfidf_w2v)
train_tfidf_w2v_df.to_csv('../data/train_tfidf_w2v_df.csv')
print('finish')

print('processing test')
test_tfidf_w2v = data_w2v(tfidf_w2v,X_test['data'])
test_tfidf_w2v_df = pd.DataFrame(data = test_tfidf_w2v)
test_tfidf_w2v_df.to_csv('../data/test_tfidf_w2v_df.csv')
print('finish')

