import gensim
import pandas as pd
import get_train_test as dataset
import json
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
import gc

stemmer = SnowballStemmer("english", ignore_stopwords = True)
def token_sent(s):
    w_list = word_tokenize(s)
    stemmed_sent = [stemmer.stem(w) for w in w_list]
    return stemmed_sent 

class StemmedTfidfCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfCountVectorizer, self).build_analyzer()
        return lambda doc:([stemmer.stem(w) for w in analyzer(doc)])


X_data = dataset.load_dataset()
X_train = X_data['train']
X_test = X_data['test']

r_data = X_train['data'] + X_test['data']
r_token_data = [token_sent(item) for item in r_data]

#r_data = X_train['data'] + X_test['data']
print('start train')
model = gensim.models.Word2Vec(r_token_data, size=200)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))
print('finish train')

print('save w2v dictionary')
model.wv.save_word2vec_format('../data/w2v_dic.txt')

r_token_data = 0
gc.collect()

print('start tfidf model training')
tfidf = StemmedTfidfCountVectorizer(stop_words = 'english', min_df = 11)
tfidf.fit(r_data)
tfidf_idf = list(tfidf.idf_)
tfidf_idf = [str(item) for item in tfidf_idf]
tfidf_dic = tfidf.vocabulary_

with open('../data/tfidf_idf_list.txt', 'w') as f:
    for item in tfidf_idf:
        f.write(item +'\n')

with open('../data/tfidf_dic.txt', 'w') as f:
    f.write(json.dumps(tfidf_dic))

