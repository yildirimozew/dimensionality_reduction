import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# The datasets are already preprocessed...
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))
"""You are given two datasets and two Python files, namely: K-meansexperiment.py and K-
medoidssexperiment.py (which already load the datasets). In these files, you are expected to
10
implement the elbow method with both K-means loss and silhouette score for the methods separately
by employing the two datasets. To get a loss value/an average silhouette score for a particular K
value (for both methods), you can run the algorithms ten times (with the same K values) and pick
the lowest loss value/average silhouette value as the result. To further alleviate randomness, this
procedure can be repeated ten times from scratch to obtain an average loss value/silhouette score
for each K value (similarly, a confidence interval can be calculated with these scores if need be).
Depending on these average loss values/silhouette scores, the most suitable K values could be picked
for both datasets with both methods. For instance, let’s say we have two datasets, A and B; first,
we would like to consider K-means, and we would like to determine a suitable K value for A. By
starting from 2 to (say) 10 we run K-means on A. First, we run for K=2 10 times on A. Since each
time different initial cluster centers are picked, it is highly likely to get different loss values/average
silhouette values for each run. After completing ten runs, we choose the lowest loss score/average
silhouette score for K=2, say α1. We repeat the same procedure for the second time (K=2, ten
runs, picking the smallest loss value/average silhouette score) to get α2. We repeat this until we
get α10. The average loss/average silhouette value calculated from α1,α2,α3...α10 is for the loss
value/average silhouette value of K=2 on dataset A. These steps need to be repeated for the rest
of the K values (K=3, K=4, ...). With these average loss values/silhouette values (on dataset A),
a K versus loss/average silhouette graph could be drawn (it provides insight into the performance
of K-means on A), and a suitable K value for A can be determined. Similarly, the whole process
can be applied to dataset B, and K-medoids could be employed similarly for clustering. Confidence
intervals calculated via αis could also be shown in K versus loss/silhouette graphs to represent
experiment results better and provide more detailed information."""

def evaluate_kmeans(dataset, k_range, n_runs=10, n_repeats=10):
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
                # Initialize and fit KMeans
                kmeans = KMeans(n_clusters=k, init="random", n_init=10, random_state=None)
                labels = kmeans.fit_predict(dataset)
                
                # Calculate loss (inertia is the sum of squared distances)
                loss = kmeans.inertia_
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
losses1, silhouettes1, loss_cis1, silhouette_cis1 = evaluate_kmeans(dataset1, k_values)

# Evaluate on dataset2
losses2, silhouettes2, loss_cis2, silhouette_cis2 = evaluate_kmeans(dataset2, k_values)

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
plot_single_metric(k_values, losses1, loss_cis1, "Loss", 'tab:red', "Dataset 1 - Kmeans (Loss)")

# Plot Silhouette Score for Dataset 1
plot_single_metric(k_values, silhouettes1, silhouette_cis1, "Silhouette Score", 'tab:blue', "Dataset 1 - Kmeans (Silhouette Score)")

# Plot Loss for Dataset 2
plot_single_metric(k_values, losses2, loss_cis2, "Loss", 'tab:red', "Dataset 2 - Kmeans (Loss)")

# Plot Silhouette Score for Dataset 2
plot_single_metric(k_values, silhouettes2, silhouette_cis2, "Silhouette Score", 'tab:blue', "Dataset 2 - Kmeans (Silhouette Score)")