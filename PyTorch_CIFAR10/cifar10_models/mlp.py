import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        in_dim=784, # MNIST
        num_classes=10,
        out_channels=[10000, 512, 64],
        split_layer=-1,
        bottleneck_dim=-1,
    ):
        super(MLP, self).__init__()
        layers = []
        in_channel = in_dim
        for out_channel in out_channels:
            # TODO: TMP for simple debugging
            layers.append(nn.Linear(in_channel, out_channel, bias=False))
            layers.append(nn.ReLU(inplace=True))
            in_channel = out_channel
        layers.append(nn.Linear(in_channel, num_classes))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        self.split_layer = split_layer
        self.prune_p = 0.0

    def set_split_layer(self, layer):
        self.split_layer = layer

    def set_prune_p(self, p):
        self.prune_p = p

    '''
    def split_forward(self, x, sigma=None, return_act=False):
        # When calculating jacobian, this is called w/o the batch dim
        if len(x.shape) == 4:
            x = x.reshape([x.shape[0], -1])
        elif len(x.shape) == 3:
            x = x.reshape([1, -1])
        else:
            assert(False)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.split_layer:
                #x = F.dropout(x, p=0.99)
                if self.prune_p > 0.0:
                    assert(self.prune_mask is not None)
                    print(x.shape, self.prune_mask.shape)
                    x *= self.prune_mask
                    # Scale x
                    x = x * (1 / (1 - self.prune_p))
                if sigma is not None:
                    x = x + torch.stack([torch.normal(torch.zeros_like(x[j]), sigma[j]) for j in range(len(sigma))])
                if return_act:
                    # TODO: TMP: Try selecting top-K (50% here for testing).
                    #orig_shape = x.shape
                    #x = x.view(x.shape[0], -1)
                    #val, idx = torch.topk(x, k=int(0.5*x.shape[1]), dim=1)
                    #res = torch.zeros_like(x)
                    #res.scatter_(1, idx, val)
                    #res = res.view(*orig_shape)
                    #print("While J calc", x[0])

                    return x

                    #return res
                    #return x
                #print("No J calc", x[0])

        return x
    '''

    def forward(self, x):
        return self.forward_second(self.forward_first(x))
    
    '''
    def forward_for_jacobian(self, x):
        return self.split_forward(x, return_act=True)
    '''

    def forward_first(self, x, for_jacobian=False):
        if len(x.shape) == 4:
            x = x.reshape([x.shape[0], -1])
        elif len(x.shape) == 3:
            # For J calc
            x = x.reshape([1, -1])
        elif len(x.shape) == 2:
            # For synthetic data
            pass
        else:
            # For synthetic data J calc
            x = x.unsqueeze(0)

        for layer in self.layers[:self.split_layer + 1]:
            x = layer(x)

        if self.prune_p > 0.0:
            x = F.dropout(x, p=self.prune_p)

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

        return x

    def set_bn_training(self, training):
        pass
