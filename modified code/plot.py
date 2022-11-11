import pickle
import matplotlib.pyplot as plt
from numpy import *

plt.figure(figsize=(10,10))
random_sample = True
skip = 16
test_ap=[]
nn_test_ap=[]
time=[]
bs=200
dataset = ['reddit', 'mooc', 'lastfm']
for i in range(10):
    # results_path = "./results/tgn-attn_{}_ramdon_sample{}/result_{}.pkl".format(random_sample, skip, i)
    results_path = "./results/tgn-attn_bs_{}/result_{}.pkl".format(bs, i)
    # results_path = "./results/tgn-attn_{}/result_{}.pkl".format(dataset[0], i)
    # results_path = "./results/tgn-attn_{}.pkl".format(i) if i>0 else "./results/tgn-attn.pkl"
    f = open(results_path, 'rb')
    content = pickle.load(f)
    print(content.keys())
    # print(content['test_ap'])
    # print(content['test_f1'])
    # print(content['test_acc'])
    # print(content['test_spe'])
    # print(content['test_sen'])
    # print('--------------------')

    test_ap.append(content['test_ap'])
    nn_test_ap.append(content['new_node_test_ap'])
    print(mean(content['epoch_times']))
    time.append(mean(content['epoch_times']))
    train_losses=content['train_losses']
    val_aps=content['val_aps']
    nn_val_aps=content['new_nodes_val_aps']
    plt.subplot(5, 2, i+1)

    # plt.plot(range(len(train_losses)), train_losses, 'r', label='train_loss')
    # plt.plot(range(len(val_aps)),val_aps, 'g', label='val_aps')
    plt.plot(range(len(nn_val_aps)), nn_val_aps, 'b', label='new_nodes_val_aps')
print('--------------------------')
print(mean(test_ap))
print('~~~~~~~~')
print(mean(nn_test_ap))
print(mean(time))
plt.tight_layout()
plt.show()





