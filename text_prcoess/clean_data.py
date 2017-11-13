import pandas as pd
import re

#data = pd.read_csv('data/step1_data.csv',index_col = 0)

def process_sentence(s):
        s = re.sub(r' \.', ' ',s)
        s = re.sub(r'#+',' ',s)
        s = re.sub(r'\*+',' ',s)
        s = re.sub(r'_+',' ',s)
        s = re.sub(r':+',' ',s)
        s = re.sub(r'\(+',' ',s)
        s = re.sub(r'\)+',' ',s)
        s = re.sub(r'\|+',' ',s)
        s = re.sub(r'\\\w+',' ',s)
        s = re.sub(r'/+',' ',s)
        s = re.sub(r'\\+',' ',s)
        s = re.sub(r'[^\x00-\x7f]',' ', s)
        #s = re.sub(r'[[:digit:]]',' ',s)
        s = re.sub(r'\s+',' ',s)
        return s

file_path = '../../common/texts/texts_train.csv'
data = pd.read_csv(file_path, index_col = 0)
 
data['recipe'] = data['recipe'].apply(lambda x: x if type(x) == str else 'remove')
data['recipe'] = data['recipe'].apply(lambda x: x if len(x) > 20 else 'r')
data = data[data['recipe'] != 'r']


data['recipe'] = data['recipe'].apply(process_sentence)
data.to_csv(file_path)
'''
data = pd.read_csv('../train_df.csv',index_col = 0)

data['recipe'] = data['recipe'].apply(process_sentence)
data.to_csv('../train_df.csv')
'''
print data.head()
