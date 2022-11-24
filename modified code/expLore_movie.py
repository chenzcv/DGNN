import pandas as pd

df = pd.read_table("./data/movie/ml-10M100K/tags.dat", sep='::', header=None, engine='python')

df.columns = (['user_id', 'item_id', "tag", "timestamp"])
user_list=[]
item_list=[]
print(df.head())
print(len(df.user_id.unique()))
print(df.user_id.max()-df.user_id.min())

print(len(df.item_id.unique()))
print(df.item_id.max()-df.item_id.min())
for index, row in df.iterrows():
    if row['user_id'] not in user_list:
        user_list.append(row['user_id'])
    if row['item_id'] not in item_list:
        item_list.append(row['item_id'])

for i,row in df.iterrows():
    df.loc[i, 'user_id'] = user_list.index(row['user_id'])
    df.loc[i, 'item_id'] = item_list.index(row['item_id'])

sort_df = df.sort_values("timestamp",inplace=True)
print(df.user_id.max())
print(len(df.user_id.unique()))
out_csv = './utils/data/movie.csv'
df.to_csv(out_csv)