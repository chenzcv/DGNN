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
