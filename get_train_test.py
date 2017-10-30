import pandas as pd

#path = '../umpc_food_101_df.csv'
def train_test_data_set():
    path = '../upmc_food_101_df.csv'
    data = pd.read_csv(path,index_col = 0)
    
    category_name = data['category'].unique()
    categories = {}
    size = len(category_name)
    for i in range(size):
        categories[category_name[i]] = i
    
    data['category_id'] = [categories[item] for item in data['category'].values]
    
    data['recipe'] = data['recipe'].apply(lambda x: x if type(x) == str else 'remove')
    data['recipe'] = data['recipe'].apply(lambda x: x if len(x) > 20 else 'r')
    data = data[data['recipe'] != 'r']

    #print(len(data.index))
    
    df_list = {'category':[],'recipe':[],'category_id':[]}
    test_df = pd.DataFrame(data = df_list)
    for i in range(size):
        test_df =  test_df.append(data[data['category_id']==i].sample(frac = 0.2))
        
    test_df = test_df[['category','category_id','recipe']]
    test_df['category_id'] = test_df['category_id'].apply(lambda x: int(x))

    data = data[['category','category_id','recipe']]
    test_index = test_df.index.values
    train_df = data.drop(test_index)

    test_df = test_df.sample(frac = 1)
    train_df = train_df.sample(frac = 1)

#    print(train_df.head())
    train_df.reset_index(inplace = True)
    train_df.drop('index',axis = 1,inplace =True)

    test_df.reset_index(inplace = True)
    test_df.drop('index',axis = 1, inplace=True)

    train_size = len(train_df.index)
    test_size = len(test_df.index)
    print 'there are {} samples in train dataset\n'.format(train_size)
 #   print(train_df.head())
    print 'there are {} samples in test dataset\n'.format(test_size)
  #  print(test_df.head())
    category_id = pd.DataFrame(data = categories,index=range(1))

    train_df.to_csv('../train_df.csv')
    test_df.to_csv('../test_df.csv')
    category_id.to_csv('../category_id.csv', index = False)

    print('finish test/train data loading\n')

def load_dataset():
    try:
        test_df = pd.read_csv('../test_df.csv')
        train_df = pd.read_csv('../train_df.csv')
        categories = pd.read_csv('../category_id.csv')
    except:
        print("Creating train/test data\n")
        train_test_data_set()
        test_df = pd.read_csv('../test_df.csv')
        train_df = pd.read_csv('../train_df.csv')
        categories = pd.read_csv('../category_id.csv')
    print('loading dataset.')
    data_dic = {}
    
    category_id = {}
    for col in categories.columns.values:
        category_id[col] = categories[col].values[0]
    
    train = {}
    train['data'] = list(train_df['recipe'].values)
    train['target'] = train_df['category_id'].values

    test = {}
    test['data'] = list(test_df['recipe'].values)
    test['target'] = test_df['category_id'].values


    data_dic['categories'] = category_id
    data_dic['train'] = train
    data_dic['test'] = test
    return data_dic
#print(size)
#print "\n"
'''print(test_df.count())
print(train_df.count())
print(test_df.head())
print(train_df.head())'''
#print(type(data['category'].values))
#print data.head(5)
#print(categories)
#load_dataset()
