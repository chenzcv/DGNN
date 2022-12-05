import pandas as pd
import time
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

f=pd.read_csv('./data/enron/emails.csv')
month=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
user_list=[]
item_list=[]
print(f.head())
for i,row in f.iterrows():
    print(row['message'])
    user = row['message'].split('\n')[2].split(': ')[1].split('@')[0]
    item = row['message'].split('\n')[3].split(': ')[1].split('@')[0]
    if user not in user_list:
        user_list.append(user)
    if item not in item_list:
        item_list.append(item)

data_dict={'user_id':[],'item_id':[],'timestamp':[]}
for i,row in f.iterrows():
    user = row['message'].split('\n')[2].split(': ')[1].split('@')[0]
    item = row['message'].split('\n')[3].split(': ')[1].split('@')[0]
    cur_time=row['message'].split('\n')[1].split(': ')[1].split('-')[0].split(', ')[1]
    cur_time=cur_time.strip()
    cur_month = cur_time.split(' ')[1]
    cur_time = cur_time.replace(cur_month, str(month.index(cur_month)+1))
    timeArray = time.strptime(cur_time, "%d %m %Y %H:%M:%S")
    timestamp = int(time.mktime(timeArray))
    data_dict['user_id'].append(user_list.index(user))
    data_dict['item_id'].append(item_list.index(item))
    data_dict['timestamp'].append(timestamp)

data_pd = pd.DataFrame(data_dict)
data_pd=data_pd.sort_values(by=['timestamp'])

print(data_pd.head())
print(data_pd.user_id.max()-data_pd.user_id.min())
print(len(data_pd.user_id.unique()))
out_csv = './utils/data/email.csv'
data_pd.to_csv(out_csv)