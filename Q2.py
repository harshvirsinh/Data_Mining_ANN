import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def read_data_set():
    dataset1 = pd.read_csv('Dataset1.csv')
    dataset2 = pd.read_csv('Dataset2.csv')
    dataset1 = dataset1.values
    dataset2 = dataset2.values
    return dataset1, dataset2


def start():
    # 1: loading datasets
    datset1, dataset2 = read_data_set()


def finding_best_k_for_clustering(min_k, max_k):
    """
    This function runs k time the k-means algorithm
    :return:
    """
    print("Finding clustering")
    seed = 7
    wcss = []
    for i in range(min_k, max_k):
        kmeans = cluster_data_with_specific_k(i, 300, 2, wl.documents_terms_vectors)
        wcss.append(kmeans.inertia_)
        print(str(i) + "," + str(kmeans.inertia_))
        stop = timeit.default_timer()
        print('Time: ', stop - start)
    calculate_best_k_clustering(wcss)


def calculate_best_k_clustering(wcss, min_k, max_k):
    """
    Finds the best k based on MSS
    :param wcss:
    :return:
    """
    x = range(min_k, max_k + len(wcss))
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
    plt.plot(range(config.min_k, config.min_k + len(wcss)), wcss)

    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    print("Errors: ")
    print(wcss)


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
