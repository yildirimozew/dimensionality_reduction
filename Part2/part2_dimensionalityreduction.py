import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
from pca import PCA
from autoencoder import AutoEncoder
import torch

# Load datasets
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

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
    pca = PCA(projection_dim=2)
    pca.fit(dataset)
    pca_transformed = pca.transform(dataset)
    return pca_transformed

def apply_autoencoder(dataset):
    dataset_tensor = torch.tensor(dataset).float()
    autoencoder = AutoEncoder(input_dim=dataset.shape[1], projection_dim=2, learning_rate=0.001, iteration_count=1500)
    autoencoder.fit(dataset_tensor) 
    return autoencoder.transform(dataset_tensor)

tsne = TSNE()
umapp = umap.UMAP()
#dataset1
print("processing Dataset 1")
pca_result = apply_pca(dataset1)
plot_2d(pca_result, "dataset1 PCA")

autoencoder_result = apply_autoencoder(dataset1)
plot_2d(autoencoder_result, "dataset1 autoencoder")

tsne_result = tsne.fit_transform(dataset1)
plot_2d(tsne_result, "dataset1 tsne")

umap_result = umapp.fit_transform(dataset1)
plot_2d(umap_result, "dataset1 umap")

#dataset2
print("processing Dataset 2")
pca_result = apply_pca(dataset2)
plot_2d(pca_result, "dataset2 PCA")

autoencoder_result = apply_autoencoder(dataset2)
plot_2d(autoencoder_result, "dataset2 autoencoder")

tsne_result = tsne.fit_transform(dataset2)
plot_2d(tsne_result, "dataset2 tsne")

umap_result = umapp.fit_transform(dataset2)
plot_2d(umap_result, "dataset2 umap")