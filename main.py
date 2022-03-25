from distutils.file_util import write_file
import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Run random forrest with specified input arguments')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=2)
    parser.add_argument('--data', type=str, default='data.csv',
                        help='data path')

    a = parser.parse_args()
    return(a.n_clusters, a.data)

def read_data(data_path):
    return anndata.read_csv(data_path)

def preprocess_data(adata: anndata.AnnData, scale :bool=True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)

def main():
    n_classifiers, data_path = parse_args()
    heart = read_data(data_path)
    #heart = read_data('data.csv')
    heart = preprocess_data(heart)
    X = PCA(heart.X, 100)
    #ran = KMeans(n_classifiers,'random',300)
    #kmeanspp = KMeans(n_classifiers,'kmeans++',300)

    save = []
    fig = plt.figure(1)
    for i in range(2,10):
        temp = KMeans(i,'random',300)
        clustering = temp.fit(X)
        t = temp.silhouette(clustering,X)
        save.append(sum(t)/len(t))
        print(i, save[-1])
        X = PCA(heart.X, 2)
        a = fig.add_subplot(4,2,i-1)
        a.title.set_text('n_clusters = %s'%(i))
        visualize_cluster(X[:,0],X[:,1],clustering)
    plt.show()

    x = [2,3,4,5,6,7,8,9]
    silhouette_plot(x,save,"Average Silhouette Coefficient Score of Clusterings (RANDOM)")
    #silhouette_plot(x,save,"Average Silhouette Coefficient Score of Clusterings (KMeans++)")
   
  
#used for plotting the average silhouette coefficient of clusterings
def silhouette_plot(x,y,title):
    plt.title(title)
    plt.ylabel("Silhouette Coefficient")
    plt.xlabel("Number of Clusters")
    plt.plot(x,y,'bo--')
    plt.show()
    

def visualize_cluster(x, y, clustering):

    num = len(np.unique(clustering))
    coloring = (255//num)*clustering
    
    plt.scatter(x,y, c = coloring)
    
   

if __name__ == '__main__':
    main()
