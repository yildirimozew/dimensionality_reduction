import pickle
from Distance import Distance
from Knn import KNN
from sklearn.model_selection import StratifiedKFold
import numpy as np

# the data is already preprocessed
dataset, labels = pickle.load(open("../datasets/part1_dataset.data", "rb"))

"""You are provided with a classification problem dataset and a partial implementation that loads
the dataset (Knnexperiment.py). On this dataset, you are expected to perform 10-fold cross-
validation (with stratification) for hyperparameter tuning (via grid search). You are free to de-
termine which hyperparameter configurations to test (there should be at least five configurations).
Please repeat this cross-validation procedure five times (by shuffling the original dataset) and com-
pute confidence intervals of accuracy performance scores for each hyperparameter configuration.
In your report, please specify these hyperparameter values and attained confidence intervals for
each hyperparameter configuration. In addition, please add some comments on how you have
picked the best-performing hyperparameter values. For stratified cross-validation, you may refer to
the following Scikitlearn documentation [11]. For data shuffling with numpy, you may refer to [12]."""

# hyperparameters
k_values = [1, 3, 5]
distance_metrics = [Distance.calculateCosineDistance, Distance.calculateMinkowskiDistance]
# 10-fold cross-validation initialization
skf = StratifiedKFold(n_splits=10)
results = []
for i in range(5):
    # shuffle the data
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    dataset = dataset[indices]
    labels = labels[indices]
    for k in k_values:
        for distance_metric in distance_metrics:
            total_acc = 0
            for train_index, test_index in skf.split(dataset, labels):
                # split the data and initialize the knn
                train_data, test_data = dataset[train_index], dataset[test_index]
                train_labels, test_labels = labels[train_index], labels[test_index]
                knn = KNN(train_data, train_labels, distance_metric, None, k)
                # predict for all test data points and calculate the accuracy
                for i in range(len(test_data)):
                    predicted_class = knn.predict(test_data[i])
                    if predicted_class == test_labels[i]:
                        total_acc += 1
            accuracy = total_acc / len(labels)
            results.append(accuracy)
    
#calculate averages
averages = []
for i in range(len(k_values)*len(distance_metrics)):
    average = sum(results[i::len(k_values)*len(distance_metrics)]) / 5
    averages.append(average)

# calculate the %95 confidence intervals
confidence_intervals = []
for i in range(len(k_values)*len(distance_metrics)):
    mean = averages[i]
    std = np.std(results[i::len(k_values)*len(distance_metrics)])
    confidence_interval = 1.96 * (std / np.sqrt(5))
    confidence_intervals.append((mean - confidence_interval, mean + confidence_interval))

# print the results
i = 0
for k in k_values:
    for distance_metric in distance_metrics:
        print("k: %d, metric: %s, confidence-: %.3f, confidence+: %.3f, avg: %.3f" % (k, distance_metric.__name__, confidence_intervals[i][0], confidence_intervals[i][1], averages[i]))
        i += 1