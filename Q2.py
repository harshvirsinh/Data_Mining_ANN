import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def read_data_set():
    dataset1 = pd.read_csv('Dataset1.csv')
    dataset2 = pd.read_csv('Dataset2.csv')
    dataset1 = dataset1.values
    dataset2 = dataset2.values
    return dataset1, dataset2


def start():
    # 1: loading datasets
    dataset1, dataset2 = read_data_set()

    # 2: clustering data with kmeans
    cluster_dataset_with_kmeans(dataset=dataset1, min_k=1, max_k=50)
    cluster_dataset_with_kmeans(dataset=dataset2, min_k=1, max_k=50)

    # 3: clustering data with dbscan
    preprocessing_data_for_dbscan(dataset1)
    preprocessing_data_for_dbscan(dataset2)
    cluster_dataset_with_dbscan(dataset=dataset1, eps=0.13, min_samples=5)
    cluster_dataset_with_dbscan(dataset=dataset2, eps=1.1, min_samples=4)

    # 4: dendrogram
    cluster_data_with_dendogram(dataset1)
    cluster_data_with_dendogram(dataset2)


def cluster_dataset_with_kmeans(dataset, min_k, max_k):
    # 1: finding best k for dataset
    k = finding_best_k_for_clustering(min_k, max_k, dataset)

    # 2: cluster dataset
    kmeans = cluster_data_with_specific_k(k, 300, 5, dataset)

    # 3: plot dataset
    plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()


def finding_best_k_for_clustering(min_k, max_k, data):
    """
    This function runs k time the k-means algorithm
    :return:
    """
    print("Finding clustering")
    seed = 7
    wcss = []
    for i in range(min_k, max_k):
        kmeans = cluster_data_with_specific_k(i, 300, 5, data)
        wcss.append(kmeans.inertia_)
        print(str(i) + "," + str(kmeans.inertia_))
    return calculate_best_k_clustering(wcss, min_k, max_k)


def calculate_best_k_clustering(wcss, min_k, max_k):
    """
    Finds the best k based on MSS
    :param max_k:
    :param min_k:
    :param wcss:
    :return:
    """
    x = range(min_k, min_k + len(wcss))
    y = wcss
    sensitivity = [1, 3, 5, 10, 100, 200, 400]
    knees = []
    norm_knees = []
    for s in sensitivity:
        kl = KneeLocator(x, y, curve='convex', direction='decreasing', S=s)
        knees.append(kl.knee)
        norm_knees.append(kl.norm_knee)

    print("knees")
    print(knees)
    plt.plot(range(min_k, min_k + len(wcss)), wcss)

    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    print("Errors: ")
    print(wcss)
    return knees[0]


def cluster_data_with_specific_k(k, max_iteration, n_init, data):
    """
    This function will clusters the data
    :param k:
    :param max_iteration:
    :param n_init:
    :param data:
    :return:
    """
    print("Clustering with K = " + str(k) + ", Started")
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=max_iteration, n_init=n_init, random_state=0)
    kmeans.fit(data)
    print("Clustering with K = " + str(k) + " Finished")
    return kmeans


def cluster_dataset_with_dbscan(dataset, eps, min_samples):
    # train
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dataset)

    clusters = clustering.labels_
    colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod',
              'lightcyan', 'navy']
    vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
    plt.scatter(dataset[:, 0], dataset[:, 1], c=vectorizer(clusters))
    plt.show()


def preprocessing_data_for_dbscan(dataset):
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(dataset)
    distances, indices = nbrs.kneighbors(dataset)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.show()


def cluster_data_with_dendogram(dataset):
    # ward is a method to calculate distance
    Z = linkage(dataset, 'ward')

    fancy_dendrogram(
        Z,
        truncate_mode='lastp',
        p=12,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=10,  # useful in small plots so annotations don't overlap
    )
    plt.show()

    # finding k
    k = find_best_k_dendrogram(Z)

    cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    cluster.fit_predict(dataset)

    plt.scatter(dataset[:, 0], dataset[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.show()


def find_best_k_dendrogram(Z):
    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    print("Dendrogram clustering found k : " + str(k))
    return k


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata
