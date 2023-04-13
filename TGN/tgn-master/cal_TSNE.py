import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def plot_statics(data, split):
    fig = plt.figure()
    plt.plot(range(len(data)), data, 'b', label=split)
    plt.title(split)
    plt.savefig('./figures/wiki_{}_memory.jpg'.format(split))
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


def calculate_TSNE(h,n=2):
    tsne = TSNE(n_components=n, init='pca', random_state=0)
    # result = tsne.fit_transform(h.cpu().detach().numpy())
    result = tsne.fit_transform(h)
    return result

def pca(data, n):
    data = np.array(data)

    # 均值
    mean_vector = np.mean(data, axis=0)

    # 协方差
    cov_mat = np.cov(data - mean_vector, rowvar=0)

    # 特征值 特征向量
    fvalue, fvector = np.linalg.eig(cov_mat)

    # 排序
    fvaluesort = np.argsort(-fvalue)

    # 取前几大的序号
    fValueTopN = fvaluesort[:n]

    # 保留前几大的数值
    newdata = fvector[:, fValueTopN]

    new = np.dot(data, newdata)

    return new