import torch.nn as nn
import torch.nn.functional as F


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim=348, output_dim=17):
        super().__init__()
        
        self.layer_1 = nn.Linear(input_dim,128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer_1(x))
        x = self.activation(self.layer_2(x))
        return self.activation(self.layer_3(x))