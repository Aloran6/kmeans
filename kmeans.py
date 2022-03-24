from turtle import update
import numpy as np
import random as r
import math

class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        clustering = np.zeros(X.shape[0])
        while iteration < self.max_iter:
            dist_matrix = self.euclidean_distance(X,self.centroids)
            clustering = dist_matrix.argmin(axis = 1)
            self.update_centroids(clustering, X)
            iteration+=1
            # your code
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        new = []
        for c in range(self.n_clusters):
            new.append(np.mean([X[i] for i in range(len(X)) if clustering[i] == c], axis = 0))
        new = np.array(new)
        print(new)
        pass
        #your code

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        centroids_list = []
        if self.init == 'random':
            # you code
            centroids_list = X[np.random.randint(X.shape[0],size = self.n_clusters)]
            self.centroids = np.array(centroids_list)
            #print(self.centroids)
            #print(X)

        elif self.init == 'kmeans++':
            # your code
            print(X)
        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # your code
        #X1 = data, X2 = centroids
        dist = []
        for i in range(self.n_clusters):
            squared = np.square(X1 - X2[i])
            sum_squred = np.sum(squared, axis = 1)
            d = np.sqrt(sum_squred)
            dist.append(d)
        dist = np.array(dist)
        dist = np.transpose(dist)
        return dist

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        # your code
        pass