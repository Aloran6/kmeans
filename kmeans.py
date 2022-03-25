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
        self.initialize_centroids(X) #get the initial centroids
        iteration = 0
        clustering = np.zeros(X.shape[0]) #create the array for storing clusterings
        while iteration < self.max_iter:
            dist_matrix = self.euclidean_distance(X,self.centroids) #compute all distances between the objects and all centroids
            clustering = dist_matrix.argmin(axis = 1) #the minimum of the distance is the clustering number
            if self.update_centroids(clustering, X): #update centroid will return TRUE if previous iteration had the same exact centroids as the current one, 
                #meaning the clustering won't change anymore, we can break and return the clusterings
                break
            iteration+=1
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        new = []
        for c in range(self.n_clusters):
            mean = np.mean([X[i] for i in range(len(X)) if clustering[i] == c], axis = 0) #get the mean value of all belongs to the same centroid
            new.append(mean)
        new = np.array(new)
        if np.array_equal(self.centroids, new): #if there was no change between this iteration and the last, we have found our final clustering choice
            return True
        else:
            self.centroids = new.copy()
            return False

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        centroids_list = []
        if self.init == 'random':
            # you code
            centroids_list = X[np.random.randint(X.shape[0],size = self.n_clusters)] #pick n_cluster amount of centroids randomly
            self.centroids = np.array(centroids_list)
            #print(self.centroids)
            #print(X)

        elif self.init == 'kmeans++':
            # your code
            self.centroids = X[r.randint(0,len(X)-1)]
            for i in range(self.n_clusters-1):
                dist_matrix = self.euclidean_distance(X,self.centroids) #get all the distances from centroids
                min_dist = np.amin(dist_matrix,axis = 1) #distance between points and their closest centroid
                sum_dist = np.sum(min_dist) #add all of the min numbers
                
                prob = min_dist/sum_dist
                choose = np.random.choice(len(X),size = 1, p=prob) #choose 1 index based on the probability   
                self.centroids = np.vstack([self.centroids, X[choose]]) #add the chosen row into our centroids list
                
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
        for i in range(len(self.centroids)): 
            squared = np.square(X1 - X2[i])
            sum_squred = np.sum(squared, axis = 1)
            d = np.sqrt(sum_squred)
            dist.append(d)
        dist = np.array(dist)
        dist = np.transpose(dist)
        return dist

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        tot = [0]*self.n_clusters
        dist_matrix = self.euclidean_distance(X,self.centroids)
        unique, counts = np.unique(clustering,return_counts=True)
        counts_dict = dict(zip(unique, counts))
        for i in range(len(X)):
            #ao = dist_matrix[dist_matrix.argmin(axis = 1)]
            ao = np.partition(dist_matrix[i],0)[0] #distance of object to its centroid
            bo = np.partition(dist_matrix[i],1)[1] #distance to the second best centroid
            so = (bo-ao)/max(ao,bo)

            #print(ao, bo, so)
            tot[clustering[i]] += so/counts_dict[clustering[i]]
        #print(dist_matrix)
        tot = np.array(tot)
        return tot
