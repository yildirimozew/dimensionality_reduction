import numpy as np
import torch.nn as nn
import torch

class AutoEncoderNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        """
        Your autoencoder model definition should go here
        """
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),  
            nn.ReLU(),
            nn.Linear(128, 2)   
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 128),  
            nn.ReLU(),
            nn.Linear(128, input_dim)  
        )
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function should map a given data matrix onto the bottleneck hidden layer
        :param x: the input data matrix of type torch.Tensor
        :return: the resulting projected data matrix of type torch.Tensor
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Your autoencoder model's forward pass operations should go here
        :param x: the input data matrix of type torch array
        :return: the neural network output as torch array
        """
        encoded = self.encoder(x)
        identity = self.decoder(encoded)
        return identity

class AutoEncoder:

    def __init__(self, input_dim: int, projection_dim: int, learning_rate: float, iteration_count: int):
        """
        Initializes the Auto Encoder method
        :param input_dim: the input data space dimensionality
        :param projection_dim: the projection space dimensionality
        :param learning_rate: the learning rate for the auto encoder neural network training
        :param iteration_count: the number epoch for the neural network training
        """
        self.input_dim = input_dim
        self.projection_matrix = projection_dim
        self.iteration_count = iteration_count
        self.autoencoder_model = AutoEncoderNetwork(input_dim, projection_dim)

        self.optimizer = torch.optim.Adam(self.autoencoder_model.parameters(), lr=learning_rate)
        #mean squared error for reconstruction loss
        self.criterion = nn.MSELoss()
        """
            Your optimizer and loss definitions should go here
        """

    def fit(self, x: torch.Tensor) -> None:
        """
        Trains the auto encoder nn on the given data matrix
        :param x: the data matrix on which the PCA is applied
        :return: None

        this function should train the auto encoder to minimize the reconstruction error
        please do not forget to put the neural network model into the training mode before training
        """
        self.autoencoder_model.train()

        for epoch in range(self.iteration_count):
            reconstructed = self.autoencoder_model(x)
            #compute loss
            loss = self.criterion(reconstructed, x)  
            #backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #output loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.iteration_count}], Loss: {loss.item():.4f}")
        
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        After training the nn a given dataset,
        this function uses the learned model to project new data instances
        :param x: the data matrix which the projection is applied on
        :return: transformed (projected) data instances (projected data matrix)
        please do not forget to put the neural network model into the evaluation mode before projecting data instances
        """
        self.autoencoder_model.eval()  
        with torch.no_grad():  
            projected = self.autoencoder_model.project(x)
        return projected

