import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import *
from imblearn.metrics import specificity_score, sensitivity_score
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap


def visualize_attention_weight(attention_weight, split):
    attention_weight = attention_weight.cpu().detach().numpy()
    cmap = sns.cubehelix_palette(rot=-0.1, gamma=0.8, light=0.4, n_colors=14)
    plot = sns.heatmap(attention_weight, xticklabels=True, cmap=cmap)
    plot.tick_params(labelsize=7)
    plt.tight_layout()
    plt.show()
    # plt.savefig('./figures2/attention_weight_{}.jpg'.format(split))


def plot_embedding_2D(data, title, split, neighbor=None, classifier='KNN'):
    size = data.shape[0]
    size //= 2
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    # colors = np.linspace(0.0, 1.0, num=len(data[:, 0]))
    # plt.scatter(data[:, 0], data[:, 1], c=colors, cmap='afmhot_r')

    y1 = np.ones(size)
    y0 = np.zeros(size)
    label = np.concatenate((y1, y0), axis=0)
    resolution = 100  # 100x100 background pixels
    X2d_xmin, X2d_xmax = np.min(data[:, 0]), np.max(data[:, 0])
    X2d_ymin, X2d_ymax = np.min(data[:, 1]), np.max(data[:, 1])
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))
    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    if classifier == 'KNN':
        background_model = KNeighborsClassifier(n_neighbors=neighbor).fit(data, label)
    elif classifier == 'DT':
        background_model = DecisionTreeClassifier(max_depth=4).fit(data, label)
    elif classifier == 'SVM':
        background_model = SVC(gamma=0.1, kernel="rbf", probability=True).fit(data, label)

    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))

    # plot
    custom_cmap = ListedColormap(['#9898ff', '#a0faa0'])
    plt.contourf(xx, yy, voronoiBackground, alpha=0.3, cmap=custom_cmap)
    s1 = plt.scatter(data[:size, 0], data[:size, 1], c='blue')
    s2 = plt.scatter(data[size:, 0], data[size:, 1], c='yellow')
    # plt.colorbar()
    plt.title(title)
    plt.legend((s1, s2), ('pos', 'neg'), loc='best')
    plt.savefig('./figures2/{}_{}.jpg'.format(split, classifier))
    return fig


def calculate_TSNE(h, split, k, neighbor=None, classifier='KNN'):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(h.cpu().detach().numpy())
    split = split + '_' + str(k)
    fig1 = plot_embedding_2D(result, 't-SNE', split, neighbor, classifier)
    plt.show()


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200, split='test',
                         negative_edge=None, use_tsne=False, visual_att=False):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    val_ap_pos, val_auc_pos = [], []
    val_ap_neg, val_auc_neg = [], []

    val_f1, val_acc = [], []
    val_spe, val_sen = [], []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        wrong_set_list = []

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

            size = len(sources_batch)
            if negative_edge is not None:
                negative_samples = np.random.choice(negative_edge, size, replace=False)
            else:
                _, negative_samples = negative_edge_sampler.sample(size)

            if split == 'small' or split == 'large' or split == 'avg':
                model.use_memory = False
                model.embedding_module.use_memory = False
            pos_prob, neg_prob, h, attention_weight = model.compute_edge_probabilities(sources_batch,
                                                                                       destinations_batch,
                                                                                       negative_samples,
                                                                                       timestamps_batch,
                                                                                       edge_idxs_batch, n_neighbors)
            if split == 'small' or split == 'large' or split == 'avg':
                model.use_memory = True
                model.embedding_module.use_memory = True
            if split == 'test' or split == 'test_unseen':
                if visual_att:
                    visualize_attention_weight(attention_weight, split + '_' + str(k))
                if use_tsne:
                    calculate_TSNE(h, split, k, classifier='DT')
                    calculate_TSNE(h, split, k, classifier='SVM')
                    calculate_TSNE(h, split, k, neighbor=3)
                    calculate_TSNE(h, split, k, neighbor=5)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            pos_wrong_index = np.where(pos_prob.cpu().numpy() > 0.5)[0]
            neg_wrong_index = np.where(neg_prob.cpu().numpy() < 0.5)[0]

            # pos_wrong_index = neg_wrong_index

            sources_wrong = sources_batch[pos_wrong_index]
            destinations_wrong = destinations_batch[pos_wrong_index]
            timestamps_wrong = timestamps_batch[pos_wrong_index]
            edge_idxs_wrong = edge_idxs_batch[pos_wrong_index]
            wrong_set = np.concatenate((sources_wrong[:, np.newaxis], destinations_wrong[:, np.newaxis],
                                        timestamps_wrong[:, np.newaxis], edge_idxs_wrong[:, np.newaxis]), axis=1)

            wrong_set_list.append(wrong_set.tolist())

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
            pos_prob = [int(item > 0.5) for item in pos_prob.cpu().numpy()]
            neg_prob = [int(item > 0.5) for item in neg_prob.cpu().numpy()]
            correct_pos = np.equal(pos_prob, np.ones(size))
            correct_neg = np.equal(neg_prob, np.zeros(size))
            val_ap_pos.append(np.mean(correct_pos))
            val_ap_neg.append(np.mean(correct_neg))

            if split == 'small' or split == 'large' or split == 'avg':
                pos_prob = [int(item > 0.5) for item in pos_prob.cpu().numpy()]
                neg_prob = [int(item > 0.5) for item in neg_prob.cpu().numpy()]
                correct_pos = np.equal(pos_prob, np.ones(size))
                correct_neg = np.equal(neg_prob, np.zeros(size))
                val_ap_pos.append(np.mean(correct_pos))
                val_ap_neg.append(np.mean(correct_neg))

                return np.mean(val_ap), np.mean(val_auc), np.mean(val_ap_pos), np.mean(val_ap_neg)

            pred_score = [int(item > 0.5) for item in pred_score]
            val_f1.append(f1_score(y_true=true_label, y_pred=pred_score))
            val_acc.append(accuracy_score(y_true=true_label, y_pred=pred_score))
            val_spe.append(specificity_score(true_label, pred_score))
            val_sen.append(sensitivity_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc), np.mean(val_f1), np.mean(val_acc), np.mean(val_spe), np.mean(
        val_sen), wrong_set_list


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
    pred_prob = np.zeros(len(data.sources))
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        decoder.eval()
        tgn.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx: e_idx]
            destinations_batch = data.destinations[s_idx: e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx: e_idx]

            source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                         destinations_batch,
                                                                                         destinations_batch,
                                                                                         timestamps_batch,
                                                                                         edge_idxs_batch,
                                                                                         n_neighbors)
            pred_prob_batch = decoder(source_embedding).sigmoid()
            pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

    auc_roc = roc_auc_score(data.labels, pred_prob)
    return auc_roc
