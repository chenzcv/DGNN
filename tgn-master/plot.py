import pickle
import matplotlib.pyplot as plt
from numpy import *

# plt.figure(figsize=(10,10))
random_sample = True
skip = 16
test_ap=[]
nn_test_ap=[]
time=[]
bs=500
dataset = ['reddit', 'mooc', 'lastfm']
for i in range(2):
    # results_path = "./results/tgn-attn_{}_ramdon_sample{}/result_{}.pkl".format(random_sample, skip, i)
    # results_path = "./results/tgn-attn_bs_{}/result_{}.pkl".format(bs, i)
    results_path = "./results/tgn-attn_{}/result_{}.pkl".format(dataset[0], i)
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
# plt.tight_layout()
# plt.show()

plt.figure()
# plt.plot([10,50,100,200,400,500,1000,2000,4000], [788.04,278,110.89,67.72,39.84,37.98,21.94,18.55,14.5], 'ro-', label='epoch time')
plt.plot([10,50,100,200,400,500,1000,2000,4000], [0.9821,0.9847,0.9825,0.9846,0.9825,0.9787,0.9818,0.9778,0.975], 'bo-', label='train_ap')
plt.plot([10,50,100,200,400,500,1000,2000,4000], [0.9729,0.9774,0.9754,0.9781,0.9763,0.9731,0.976,0.9724,0.9691], 'go-', label='nn_train_ap')
plt.legend()
plt.show()




