Dataset: http://visiir.lip6.fr/data/public/UPMC_Food101.tar.gz
Related Paper: http://visiir.univ-lr.fr/images/publication/CEA_ICME2015.pdf

1.clean data
1)remove meaningless marks such as |, #, _, ! and extra whitespace
2) remove stop words
3) stemming words

2. data processing
1) using stemmerCountVectrizer()
2) TidfTransformer()
3) store the transfered results avoid converting data from scratch for saving time

3. use serial methods
1) try mnb() with default setting
2) try SVM() with SVC() defaault setting
3) try mlp classifier

4. reduce dimension
compress the world vector 
using such as words embedding
combine tfidf and w2v method to process text data

5. classfier
nb() uses results from tfidf countvector without w2v(word embedding)
svm() uses results combining tdidf and w2v

whole procedure liek following:
step 1. read_rawdata.py -> step 2. clean_data.py -> 
step 3. get_train_test.py -> step 4. construct_dict.py + tfidf_w2v.py ,countvector_data.py (tow different methods of processing data) -> step 5. using classifier to do classify (*clf.py)
