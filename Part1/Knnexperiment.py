import pickle
from Distance import Distance
from Knn import KNN
from sklearn.model_selection import StratifiedKFold
import numpy as np

# the data is already preprocessed
dataset, labels = pickle.load(open("../datasets/part1_dataset.data", "rb"))

# hyperparameters
k_values = [1, 3, 5]
distance_metrics = [Distance.calculateCosineDistance, Distance.calculateMinkowskiDistance]
#10 split cross validation
skf = StratifiedKFold(n_splits=10)
results = []
for i in range(5):
    # data shuffling
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    dataset = dataset[indices]
    labels = labels[indices]
    for k in k_values:
        for distance_metric in distance_metrics:
            total_acc = 0
            for train_index, test_index in skf.split(dataset, labels):
                #split the data
                train_data, test_data = dataset[train_index], dataset[test_index]
                train_labels, test_labels = labels[train_index], labels[test_index]
                knn = KNN(train_data, train_labels, distance_metric, None, k)
                # predict and calculate accuracy
                for i in range(len(test_data)):
                    predicted_class = knn.predict(test_data[i])
                    if predicted_class == test_labels[i]:
                        total_acc += 1
            accuracy = total_acc / len(labels)
            results.append(accuracy)
    

averages = []
for i in range(len(k_values)*len(distance_metrics)):
    #take the sums of results with step of len(k_values)*len(distance_metrics), which are the results of the same k and distance metric
    average = sum(results[i::len(k_values)*len(distance_metrics)]) / 5
    averages.append(average)

# calculate the %95 confidence intervals
confidence_intervals = []
for i in range(len(k_values)*len(distance_metrics)):
    mean = averages[i]
    #step of len(k_values)*len(distance_metrics) to take the results of the same k and distance metric
    std = np.std(results[i::len(k_values)*len(distance_metrics)])
    confidence_interval = 1.96 * (std / np.sqrt(5))
    confidence_intervals.append((mean - confidence_interval, mean + confidence_interval))

#results
i = 0
for k in k_values:
    for distance_metric in distance_metrics:
        print("k: %d, metric: %s, confidence-: %.3f, confidence+: %.3f, avg: %.3f" % (k, distance_metric.__name__, confidence_intervals[i][0], confidence_intervals[i][1], averages[i]))
        i += 1