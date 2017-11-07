1.clean data
1)remove meaningless marks such as |, #, _, ! and extra whitespace
2) remove stop words
3) stemming words

2. data processing
1) using stemmerCountVectrizer()
2) TidfTransformer()
3) store the transfered results avoid converting data from scratch for saving time

3. use serial methods
1) GridSearvchCv() doesn't work with memory problems
2) try mnb() with default setting
3) try SVM() with SVC() defaault setting
4) try mlp classifier

next work and find a proper method for classifier
4. reduce dimension
compress the world vector 
using such as words embedding
combine tfidf and w2v method to process text data


read_rawdata.py -> clean_data.py -> get_train_test.py -> construct_dict.py,countvector_data.py ->*clf.py 
           						 (tow different methods of processing data)
