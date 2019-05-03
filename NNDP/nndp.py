import torch
import torch.nn as nn

class NNDP(nn.Module):
    '''
        Creates the neural network dynamic programming model given 
        the number of verticies in the graph

        params:
            n: the number of verticies in the graph
    '''
    def __init__(self, n, out_features= 1):
        super(NNDP, self).__init__()
        self.out_features = out_features
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
        layers += [nn.Linear(4*n, self.out_features)]
        layers += [nn.Softmax(dim=0)]
        
        return nn.Sequential(*layers)