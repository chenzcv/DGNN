# DGNN
Run original experiment

```
Wikipedia:
python train_self_supervised.py --use_memory --prefix tgn-attn --n_runs 10

Reddit:
python train_self_supervised.py -d reddit --use_memory --prefix tgn-attn --n_runs 10
```
Partition the dataset into two datasets D1, D2 by sampling two data points from every 2/4/8/16/32 data points

```
D1:
python train_tbatch.py --use_memory --prefix tgn-attn --sample_type uniform --skip 16 --id 0

D2:
python train_tbatch.py --use_memory --prefix tgn-attn --sample_type uniform --skip 16 --id 1
```

Run the training and average the parameters of M1 and M2 for every iteration

```
python train_sample.py --use_memory --prefix tgn-attn --sample_type uniform --skip 16
```

Divide the dataset into m t-batch, for each epoch, randomly sampled k t-batch (k<m)

```
python train_tbatch.py --use_memory --prefix tgn-attn --sample_type t-batch --skip 4 --id 0
```
# Data Exploration (Download Raw Data)
MOOC

```
wget http://snap.stanford.edu/jodie/mooc.csv
```
LastFM

```
wget http://snap.stanford.edu/jodie/lastfm.csv
```
Autonomous systems AS-733

```
wget http://snap.stanford.edu/data/as20000102.txt.gz
```

MovieLens-10M (or directly download the 'tags.dat' in file /data/movie)

```
wget https://files.grouplens.org/datasets/movielens/ml-10m.zip
```

FB-Forum (or directly download the 'fb-forum.edges' in file /data/fb)

```
wget https://nrvis.com/download/data/dynamic/fb-forum.zip
```

Enron-Email
Download Link: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
