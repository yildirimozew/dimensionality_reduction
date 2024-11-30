import numpy as np
import torch.nn as nn
import torch

"""Similarly, you are provided a file named autoencoder.py where you are expected to implement the
autoencoder method using solely numpy and torch libraries. In part2 dimensionalityreduction.py,
you are expected to import this file for performing data projection via the autoencoder method. 
The AutoencoderNetwork class should implement a regression MLP architecture where the number
of inputs is equal to the number of outputs and one of the hidden layers (bottleneck layer) should
have two nodes in order to perform a projection from the original input data space to 2-D. The
neural network should approximate the identity function (f(x) = x). In other words, the network
should learn to generate the input data instances at its output layer as closely as possible. To this
end, it can be trained with the reconstruction loss. For the AutoencoderNetwork class, you are expected to first define your
architecture in the constructor, the structure of the neural network is up to you (it should contain
at least one hidden layer, which is the bottleneck layer). The forward function should implement
the ordinary regression network forward pass operations. The project function should return the
output of the bottleneck layer of the network. """

"""For the autoencoder architecture, in
your implementation, you can consider a single fixed set of hyperparameters (learning rate, iteration
count, number of hidden layers, activation functions utilized, optimizer utilized) for projecting
datasets (due to the time constraints of the assignment)."""

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
