import numpy as np

class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def predict(self, instance):
        distances = np.array([self.similarity_function(instance, x, self.similarity_function_parameters) for x in self.dataset])
        sorted_indexes = np.argsort(distances)
        k_nearest_labels = self.dataset_label[sorted_indexes[:self.K]]
        return np.argmax(np.bincount(k_nearest_labels))


