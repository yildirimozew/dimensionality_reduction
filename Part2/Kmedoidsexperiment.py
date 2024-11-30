import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

# The datasets are already preprocessed...
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

def evaluate_kmedoids(dataset, k_range, n_runs=10, n_repeats=10):
    average_losses = []
    average_silhouettes = []
    loss_confidence_intervals = []
    silhouette_confidence_intervals = []
    for k in k_range:
        all_losses = []
        all_silhouettes = []
        for n in range(n_repeats):
            run_losses = []
            run_silhouettes = []
            for _ in range(n_runs):
                #initialize and fit kmedoids
                kmedoids = KMedoids(n_clusters=k, init="random", random_state=None)
                labels = kmedoids.fit_predict(dataset)
                #calculate loss
                loss = kmedoids.inertia_
                run_losses.append(loss)
                #calculate silhouette, requires at least 2 clusters
                if len(np.unique(labels)) > 1:  
                    silhouette = silhouette_score(dataset, labels)
                    run_silhouettes.append(silhouette)
            #pick best values from this run
            all_losses.append(np.min(run_losses))
            if run_silhouettes:
                all_silhouettes.append(np.mean(run_silhouettes))
        # calculate averages and confidence intervals
        avg_loss = np.mean(all_losses)
        average_losses.append(avg_loss)
        loss_confidence_intervals.append(1.96 * np.std(all_losses) / np.sqrt(n_repeats))

        avg_silhouette = np.mean(all_silhouettes)
        average_silhouettes.append(avg_silhouette)
        silhouette_confidence_intervals.append(1.96 * np.std(all_silhouettes) / np.sqrt(n_repeats))
    
    return average_losses,average_silhouettes,loss_confidence_intervals, silhouette_confidence_intervals

# k range
k_values = range(2, 11)

#evaluation
losses1, silhouettes1, loss_cis1, silhouette_cis1 = evaluate_kmedoids(dataset1, k_values)
losses2, silhouettes2, loss_cis2, silhouette_cis2 = evaluate_kmedoids(dataset2, k_values)

def plot_single_metric(k_values, metric, metric_cis, ylabel, title):
    plt.figure(figsize=(8, 6))
    plt.errorbar(k_values, metric, yerr=metric_cis, label=ylabel, fmt='-o')
    plt.xlabel('K amount')
    plt.ylabel(ylabel)
    plt.tick_params(axis='y')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

#plot the results
plot_single_metric(k_values, losses1, loss_cis1, "loss", "dataset 1 - kmedoids - loss")
plot_single_metric(k_values, silhouettes1, silhouette_cis1, "silhouette value", "dataset 1 - kmedoids silhouette score)")
plot_single_metric(k_values, losses2, loss_cis2, "loss", "Dataset 2 - kmedoids - loss")
plot_single_metric(k_values, silhouettes2, silhouette_cis2, "sillhouette value", "dataset 2 - kmedoids silhouette score")