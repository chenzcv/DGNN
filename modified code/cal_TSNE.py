import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def plot_statics(data, split):
    fig = plt.figure()
    plt.plot(range(len(data)), data, 'b', label=split)
    plt.title(split)
    plt.savefig('./figures/reddit_{}_memory.jpg'.format(split))
    plt.show()


def plot_embedding_2D(data, title, split):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    # colors = np.linspace(0.0, 1.0, num=len(data[:, 0]))
    # plt.scatter(data[:, 0], data[:, 1], c=colors, cmap='afmhot_r')
    plt.scatter(data[:, 0], data[:, 1])
    # plt.colorbar()
    plt.title(title)
    plt.savefig('./figures/{}.jpg'.format(split))
    return fig


def calculate_TSNE(h, split, k):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(h.cpu().detach().numpy())
    split = split + '_' + str(k)
    fig1 = plot_embedding_2D(result, 't-SNE', split)
    plt.show()
