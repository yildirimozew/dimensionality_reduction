import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram

# Load dataset
dataset = pickle.load(open("../datasets/part3_dataset.data", "rb"))

def plot_silhouette_analysis(data, labels, title):
    silhouette_vals = silhouette_samples(data, labels)
    avg_score = np.mean(silhouette_vals)
    y_lower = 10
    for i in range(len(set(labels))):
        ith_cluster_silhouette_vals = silhouette_vals[labels == i]
        ith_cluster_silhouette_vals.sort()
        size = len(ith_cluster_silhouette_vals)
        y_upper = y_lower + size
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals, alpha=0.7)
        y_lower = y_upper + 10
    plt.title(f"{title} (Avg Silhouette Score: {avg_score:.2f})")
    plt.xlabel("silhouette coefficient")
    plt.ylabel("cluster")
    plt.show()
    return avg_score

def plot_dendrogram(model, title):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, (left, right) in enumerate(model.children_):
        counts[i] = counts[left - n_samples + 1] + counts[right - n_samples + 1] + 1
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    plt.figure(figsize=(8, 6))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=10, show_leaf_counts=True)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()

linkage_criteria = ['single', 'complete']
distance_metrics = ['euclidean', 'cosine']

for linkage_method in linkage_criteria:
    for metric in distance_metrics:
        for k in range(2, 6):
            print(f"Running HAC with linkage={linkage_method}, metric={metric}, k={k}")
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method, metric=metric, compute_distances=True)
            model = model.fit(dataset)
            avg_silhoette_score = plot_silhouette_analysis(dataset, model.labels_, f"HAC Silhoette Analysis (k={k}, linkage={linkage_method}, metric={metric})")
            print(f"Average Silhouette Score: {avg_silhoette_score}")
            plot_dendrogram(model, f"Dendrogram {linkage_method} {metric} {k}")

eps_range = [0.1, 0.4, 0.7, 1.0]
min_samples_range = [2, 5, 10, 15]
metrics = ['euclidean', 'cosine']
dbscan_results = []

for eps in eps_range:
    for min_samples in min_samples_range:
        for metric in metrics:
            print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}, metric={metric}")
            model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            labels = model.fit_predict(dataset)
            if(len(set(labels)) == 1):
                print("Only 1 cluster found.skipping")
                continue
            score = silhouette_score(dataset, labels)
            dbscan_results.append((eps, min_samples, metric, score))

# select top 4 configurations
dbscan_results = sorted(dbscan_results, key=lambda x: x[3], reverse=True)[:4]
print("Top 4 DBSCAN configurations (by silhouette score):", dbscan_results)

# silhouette analysis
for eps, min_samples, metric, _ in dbscan_results:
    print(f"Silhouette Analysis for eps={eps}, min_samples={min_samples}, metric={metric}")
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(dataset)
    plot_silhouette_analysis(dataset, labels, f"DBSCAN (eps={eps}, min_samples={min_samples}, metric={metric})")

