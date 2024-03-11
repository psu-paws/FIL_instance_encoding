import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, split_layer=-1):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
        self.split_layer = split_layer

    def set_split_layer(self, layer):
        self.split_layer = layer

    def set_prune_p(self, prune):
        pass

    def set_bn_training(self, training):
        pass

    def forward(self, x):
        return self.forward_second(self.forward_first(x))

    def forward_first(self, x, for_jacobian=True):
        for layer in self.layers[:self.split_layer + 1]:
            x = layer(x)
        return x

    def forward_second(self, x, sigma=None):
        if sigma is not None:
            if len(sigma.shape) > 0:
                noise = torch.stack([torch.normal(torch.zeros_like(x[j]), sigma[j]) for j in range(len(sigma))])
            else:
                noise = torch.normal(torch.zeros_like(x), sigma)
            x = x + noise

        for layer in self.layers[self.split_layer + 1:]:
            x = layer(x)

        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output


