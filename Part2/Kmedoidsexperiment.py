import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

# The datasets are already preprocessed...
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

def evaluate_kmedoids(dataset, k_range, n_runs=10, n_repeats=10):
    avg_losses = []
    avg_silhouettes = []
    loss_conf_intervals = []
    silhouette_conf_intervals = []

    for k in k_range:
        all_losses = []
        all_silhouettes = []
        
        for _ in range(n_repeats):
            run_losses = []
            run_silhouettes = []
            
            for _ in range(n_runs):
                # Initialize and fit Kmedoids
                kmedoids = KMedoids(n_clusters=k, init="random", random_state=None)
                labels = kmedoids.fit_predict(dataset)
                
                # Calculate loss (inertia is the sum of squared distances)
                loss = kmedoids.inertia_
                run_losses.append(loss)
                
                # Calculate silhouette score
                if len(np.unique(labels)) > 1:  # Avoid silhouette calculation error
                    silhouette = silhouette_score(dataset, labels)
                    run_silhouettes.append(silhouette)
            
            # Pick best values from this run
            all_losses.append(np.min(run_losses))  # Choose lowest loss
            if run_silhouettes:
                all_silhouettes.append(np.mean(run_silhouettes))  # Choose mean silhouette score
        
        # Calculate average and confidence intervals
        avg_loss = np.mean(all_losses)
        avg_losses.append(avg_loss)
        loss_conf_intervals.append(1.96 * np.std(all_losses) / np.sqrt(n_repeats))

        avg_silhouette = np.mean(all_silhouettes) if all_silhouettes else 0
        avg_silhouettes.append(avg_silhouette)
        silhouette_conf_intervals.append(1.96 * np.std(all_silhouettes) / np.sqrt(n_repeats) if all_silhouettes else 0)
    
    return avg_losses, avg_silhouettes, loss_conf_intervals, silhouette_conf_intervals

# Define range of K values
k_values = range(2, 11)

# Evaluate on dataset1
losses1, silhouettes1, loss_cis1, silhouette_cis1 = evaluate_kmedoids(dataset1, k_values)

# Evaluate on dataset2
losses2, silhouettes2, loss_cis2, silhouette_cis2 = evaluate_kmedoids(dataset2, k_values)

# Plot results
def plot_single_metric(k_values, metric, metric_cis, ylabel, color, title):
    plt.figure(figsize=(8, 6))
    plt.errorbar(k_values, metric, yerr=metric_cis, label=ylabel, color=color, fmt='-o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel(ylabel, color=color)
    plt.tick_params(axis='y', labelcolor=color)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Plot Loss for Dataset 1
plot_single_metric(k_values, losses1, loss_cis1, "Loss", 'tab:red', "Dataset 1 - Kmedoids (Loss)")

# Plot Silhouette Score for Dataset 1
plot_single_metric(k_values, silhouettes1, silhouette_cis1, "Silhouette Score", 'tab:blue', "Dataset 1 - Kmedoids (Silhouette Score)")

# Plot Loss for Dataset 2
plot_single_metric(k_values, losses2, loss_cis2, "Loss", 'tab:red', "Dataset 2 - Kmedoids (Loss)")

# Plot Silhouette Score for Dataset 2
plot_single_metric(k_values, silhouettes2, silhouette_cis2, "Silhouette Score", 'tab:blue', "Dataset 2 - Kmedoids (Silhouette Score)")