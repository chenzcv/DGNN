import numpy as np
import pandas as pd
from pathlib import Path
# pd.set_option('display.max_columns', None)
# data_name = ['wikipedia','reddit','mooc','lastfm']
# PATH = './utils/data/{}.csv'.format(data_name[2])
#
# data=pd.read_csv(PATH)
# print(data.info())
# print(data.head())
# with open(PATH) as f:
#     s = next(f)
#     for idx, line in enumerate(f):
#         e = line.strip().split(',')
#         u = int(e[0])  # user_id
#         i = int(e[1])  # item_id
#
#         ts = float(e[2])  # timestamp
#         label = float(e[3])  # int(e[3])  #state_label
#
#         feat = np.array([float(x) for x in e[4:]])
#         print(feat)

# f=open('./data/autosys/')

import os
import pandas as pd
import json

out_csv = './utils/data/autosys.csv'
data_pd = pd.read_csv(out_csv)
print(len(data_pd.user_id.unique()))
print(len(data_pd.item_id.unique()))
print(len(data_pd))

filePath = './data/autosys'
for i,j,k in os.walk(filePath):
    print(k)
with open('./utils/data/autosys_id.json', encoding='utf-8') as f:
    id_dict = json.load(f)
    f.close()

id_list=list(id_dict.keys())[:500]
file_list = sorted(k, key=str.lower)
print(len(file_list))
ts=0
data_dict={'user_id':[],'item_id':[],'timestamp':[],'label':[],'edge_feature':[]}
cur_id=0
for file in file_list:
    file = os.path.join(filePath,file)
    with open(file) as f:
        s = next(f)
        for idx, line in enumerate(f):
            if line[0] == '#':
                continue
            # print(line)
            item = line[:-1].split('\t')
            user_id = item[0]
            item_id = item[1]
            if item_id not in id_list:
                continue
            data_dict['user_id'].append(int(id_dict[user_id]))
            data_dict['item_id'].append(id_list.index(item_id))
            data_dict['timestamp'].append(float(ts))
            data_dict['label'].append(0)
            data_dict['edge_feature'].append(0.0)
    ts+=1

# id_list = sorted(id_list)
# id_dict={}
# for i in range(len(id_list)):
#     id_dict[id_list[i]]=i
# with open('./utils/data/autosys_id.json', 'w') as file:
#     file.write(json.dumps(id_dict))
data_pd = pd.DataFrame(data_dict)
# for i,row in data_pd.iterrows():
#     data_pd.loc[i, 'user_id'] = id_dict[row['user_id']]
#     data_pd.loc[i, 'item_id'] = id_dict[row['item_id']]
print(data_pd.head())
print(data_pd.user_id.max())
print(len(data_pd.user_id.unique()))
out_csv = './utils/data/autosys.csv'
data_pd.to_csv(out_csv)