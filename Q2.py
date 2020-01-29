import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np


def read_data_set():
    dataset1 = pd.read_csv('Dataset1.csv')
    dataset2 = pd.read_csv('Dataset2.csv')
    dataset1 = dataset1.values
    dataset2 = dataset2.values
    return dataset1, dataset2


colors = ['red', 'green', 'yellow', 'blue', 'black']


def start():
    # 1: loading datasets
    dataset1, dataset2 = read_data_set()

    # 2: clustering data with kmeans
    # cluster_dataset_with_kmeans(dataset=dataset1, min_k=1, max_k=50)
    # cluster_dataset_with_kmeans(dataset=dataset2, min_k=1, max_k=50)

    # 3: clustering data with dbscan
    preprocessing_data_for_dbscan(dataset1)
    preprocessing_data_for_dbscan(dataset2)
    cluster_dataset_with_dbscan(dataset=dataset1, eps=0.12, min_samples=5)
    cluster_dataset_with_dbscan(dataset=dataset2, eps=0.9, min_samples=5)


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

    # plot
    classes = []
    for i in range(0, clustering.components_.shape[0]):
        classes.append([])

    for index, item in enumerate(clustering.labels_):
        classes[item].append(dataset[index])

    for i in range(0, len(classes)):
        if len(classes[i]) == 0:
            continue
        plt.scatter(np.array(classes[i])[:, 0], np.array(classes[i])[:, 1], c=colors[i % 4])
        # plt.scatter(clustering.components_[i][0], clustering.components_[i][1], s=50, c=colors[i])

    plt.show()


def preprocessing_data_for_dbscan(dataset):
    min_distance = 100000000
    max_distance = 0
    average_distance = 0
    counter = 0
    for i in range(0, len(dataset) - 1):
        for j in range(i, len(dataset)):
            dist = np.linalg.norm(dataset[i] - dataset[j])
            average_distance += dist
            counter += 1
            max_distance = max(max_distance, dist)
            min_distance = min(min_distance, dist)
    average_distance /= counter

    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(dataset)
    distances, indices = nbrs.kneighbors(dataset)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.show()
    return min_distance, max_distance, average_distance
