import torch
import torch.nn as nn

class NNDP(nn.Module):
    def __init__(self, n):
        super(NNDP, self).__init__()
        self.layers = self.make_nndp_layers(n)

    def forward(self, input_data):
        x = self.layers(input_data)
        return x

    def make_nndp_layers(self, n):
        layers = []
        # input layer
        layers += [nn.Linear(2*n, 4*n)]
        # hidden layers
        layers += [nn.Linear(4*n, 4*n), nn.Sigmoid()]
        layers += [nn.Linear(4*n, 4*n), nn.Sigmoid()]
        # output layer
        layers += [nn.Linear(4*n, 1)]
        
        return nn.Sequential(*layers)