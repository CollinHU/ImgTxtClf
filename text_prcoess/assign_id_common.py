import pandas as pd


with open('../image_process/category_id.txt', 'r') as f:
            categories = eval(f.read())

df = pd.read_csv('/mnt/zyhu/common/texts/texts_train.csv',index_col = 0)
#df_1 = pd.read_csv('/mnt/zyhu/common/texts/train_tfidf_w2v_df.csv',index_col = 0)
#df_1['category_id'] = df['category_id']
df['category_id'] = [categories[item] for item in df['category'].values]
#df.to_csv('/mnt/zyhu/common/texts/train_tfidf_w2v_df.csv')
df.to_csv('/mnt/zyhu/common/texts/texts_train.csv')

df = pd.read_csv('/mnt/zyhu/common/texts/texts_val.csv',index_col = 0)
#df_1 = pd.read_csv('/mnt/zyhu/common/texts/val_tfidf_w2v_df.csv',index_col = 0)
#df_1['category_id'] = df['category_id']
df['category_id'] = [categories[item] for item in df['category'].values]
#df.to_csv('/mnt/zyhu/common/texts/val_tfidf_w2v_df.csv')
df.to_csv('/mnt/zyhu/common/texts/texts_val.csv')

df = pd.read_csv('/mnt/zyhu/common/texts/texts_test.csv',index_col = 0)
df['category_id'] = [categories[item] for item in df['category'].values]
#df_1 = pd.read_csv('/mnt/zyhu/common/texts/test_tfidf_w2v_df.csv',index_col = 0)
#df_1['category_id'] = df['category_id']
#df.to_csv('/mnt/zyhu/common/texts/test_tfidf_w2v_df.csv')
df.to_csv('/mnt/zyhu/common/texts/texts_test.csv')
