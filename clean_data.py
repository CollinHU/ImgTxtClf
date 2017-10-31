import pandas as pd
import re

#data = pd.read_csv('data/step1_data.csv',index_col = 0)

def process_sentence(s):
        s = re.sub(' \.', '.',s)
        s = re.sub('#+',' ',s)
        s = re.sub('\*+',' ',s)
        s = re.sub('_+',' ',s)
        s = re.sub('\(',' ',s)
        s = re.sub('\)',' ',s)
        s = re.sub('\|',' ',s)
        s = re.sub('\s+',' ',s)
        return s

data = pd.read_csv('../test_df.csv',index_col = 0)

data['recipe'] = data['recipe'].apply(process_sentence)
data.to_csv('../test_df.csv')

data = pd.read_csv('../train_df.csv',index_col = 0)

data['recipe'] = data['recipe'].apply(process_sentence)
data.to_csv('../train_df.csv')

print data.head()
