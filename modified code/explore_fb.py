import pandas as pd

f = open('./data/FB-forum/fb-forum.edges', 'r')
data_dict={'user_id':[],'item_id':[],'timestamp':[]}
# item_list=[]
# for idx, line in enumerate(f):
#     e = line.strip().split(',')
#     item=e[1]
#     if item not in item_list:
#         item_list.append(int(item))
for idx, line in enumerate(f):
    e = line.strip().split(',')
    data_dict['user_id'].append(int(e[0]))
    data_dict['item_id'].append(int(e[1]))
    data_dict['timestamp'].append(float(e[2]))
data_pd = pd.DataFrame(data_dict)

print(data_pd.head())
print(data_pd.user_id.max())
print(len(data_pd.user_id.unique()))
out_csv = './utils/data/fb.csv'
data_pd.to_csv(out_csv)