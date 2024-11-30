"""you are given a Python file named part3 dimensionalityreduction.py, which
already loads the dataset of this part. In this file, you are expected to employ the three dimension-
ality reduction methods (PCA, t-SNE, and UMAP) for visualization of the dataset of this part in
2-D. You can try different hyperparameter configurations for t-SNE and UMAP to enhance visu-
alization results (please ensure that the hyperparameter configurations you have tested are explicit
in the file). In the report, please provide the ”best” dimensionality reduction results (as matplotlib
scatter plots) of each method (the definition of ”best” is up to you)."""

import pickle
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# The dataset is already preprocessed...
data = pickle.load(open("../datasets/part3_dataset.data", "rb"))

# Utility function for plotting
def plot_2d(data, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.8)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# Apply dimensionality reduction methods
def apply_pca(dataset):
    pca = PCA(n_components=2)
    pca.fit(dataset)
    pca_transformed = pca.transform(dataset)
    return pca_transformed

tsne = TSNE()
umapp = umap.UMAP()

pca_result = apply_pca(data)
plot_2d(pca_result, "PCA")

tsne_result = tsne.fit_transform(data)
plot_2d(tsne_result, "tsne")

umap_result = umapp.fit_transform(data)
plot_2d(umap_result, "umap")
