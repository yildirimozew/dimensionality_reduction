import numpy as np

class PCA:
    def __init__(self, projection_dim: int):
        """
        Initializes the PCA method
        :param projection_dim: the projection space dimensionality
        """
        self.projection_dim = projection_dim
        self.projection_matrix = None

    def fit(self, x: np.ndarray) -> None:
        """
        Applies the PCA method and obtains the projection matrix
        :param x: the data matrix on which the PCA is applied
        :return: None

        this function should assign the resulting projection matrix to self.projection_matrix
        """
        mean = np.mean(x, axis=0)
        #center the data
        x_centered = x - mean
        #calculate the covariance matrix
        covariance_matrix = np.cov(x_centered, rowvar=False)
        #calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        #sort the indices of eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        #sort the eigenvalues and eigenvectors
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        #extract the projection matrix by selecting most significant eigenvectors
        self.projection_matrix = eigenvectors[:, :self.projection_dim]

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        After learning the projection matrix on a given dataset,
        this function uses the learned projection matrix to project new data instances
        :param x: data matrix which the projection is applied on
        :return: transformed (projected) data instances (projected data matrix)
        this function should utilize self.projection_matrix for the operations
        """

        mean = np.mean(x, axis=0)
        x_centered = x - mean
        return np.dot(x_centered, self.projection_matrix)