import numpy as np
import pandas as pd
from pathlib import Path
pd.set_option('display.max_columns', None)
data_name = ['wikipedia','reddit','mooc','lastfm']
PATH = './utils/data/{}.csv'.format(data_name[1])

data=pd.read_csv(PATH)
print(data.info())
print(data.head())
with open(PATH) as f:
    s = next(f)
    for idx, line in enumerate(f):
        e = line.strip().split(',')
        u = int(e[0])  # user_id
        i = int(e[1])  # item_id

        ts = float(e[2])  # timestamp
        label = float(e[3])  # int(e[3])  #state_label

        feat = np.array([float(x) for x in e[4:]])
        print(feat)