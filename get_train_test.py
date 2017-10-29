import pandas as pd

data = pd.read_csv('../upmc_food_101_df.csv',index_col = 0)

category_name = data['category'].unique()
categories = {}
df_list = {'category':[],'recipe':[],'category_id':[]}
size = len(category_name)
for i in range(size):
    categories[category_name[i]] = i

data['category_id'] = [categories[item] for item in data['category'].values]


test_df = pd.DataFrame(data = df_list)
for i in range(size):
   test_df =  test_df.append(data[data['category_id']==i].sample(frac = 0.2))

test_df = test_df[['category','category_id','recipe']]
data = data[['category','category_id','recipe']]

test_index = test_df.index.values
train_df = data.drop(test_index)
train_df.to_csv('train_df.csv')
test_df.to_csv('test_df.csv')
#print(size)
print "\n"
print(test_df.count())
print(train_df.count())
print(test_df.head())
print(train_df.head())
#print(type(data['category'].values))
#print data.head(5)
#print(categories)


