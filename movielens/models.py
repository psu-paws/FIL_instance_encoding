from torch import nn
import torch


class NCF_MLP(nn.Module):
    def __init__(self, emb_size=[], emb_dim=32, mlp_dims=[], split_layer=-1, bottleneck_dim=-1):
        super(NCF_MLP, self).__init__()
        self.embs = nn.ModuleList()
        for size in emb_size:
            self.embs.append(nn.Embedding(size, emb_dim))

        mlp = []
        in_dim = len(emb_size) * emb_dim
        for i, dim in enumerate(mlp_dims):
            mlp.append(nn.Linear(in_dim, dim))
            mlp.append(nn.ReLU())
            in_dim = dim
        mlp.append(nn.Linear(in_dim, 1))
        mlp.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*mlp)
        self.split_layer = split_layer
        self.prune_p = 0.0

        if bottleneck_dim > 0:
            in_dim = mlp_dims[split_layer // 2] # Because relu follows linear
            self.compress = nn.Sequential(
                    nn.Linear(in_dim, bottleneck_dim),
                    nn.ReLU(inplace=True))

            self.decompress = nn.Sequential(
                    nn.Linear(bottleneck_dim, in_dim),
                    nn.ReLU(inplace=True))
        else:
            self.compress = None
            self.decompress = None


    def forward(self, x):
        return self.forward_second(self.forward_first(self.forward_embs(x)))

    def forward_embs(self, x):
        embs = []
        for t, emb in zip(x, self.embs):
            e = emb(t)
            embs.append(e)

        return torch.cat(embs, dim=1)

    # We assume the output of the emb is what we want to reconstruct
    def split_forward(self, x, sigma=None, return_act=False):
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i == self.split_layer:
                if self.compress is not None:
                    x = self.compress(x)
                if sigma is not None:
                    x = x + torch.stack([torch.normal(torch.zeros_like(x[j]), sigma[j]) for j in range(len(sigma))])
                if return_act:
                    return x
                if self.compress is not None:
                    x = self.decompress(x)

        return x

    def forward_first(self, x, for_jacobian=True):
        for layer in self.mlp[:self.split_layer + 1]:
            x = layer(x)

        if self.compress is not None:
            x = self.compress(x)

        return x

    def forward_second(self, x, sigma=None):
        if sigma is not None:
            if len(sigma.shape) > 0:
                noise = torch.stack([torch.normal(torch.zeros_like(x[j]), sigma[j]) for j in range(len(sigma))])
            else:
                noise = torch.normal(torch.zeros_like(x), sigma)
            if self.prune_p > 0.0:
                x[x != 0] += noise[x != 0]
            else:
                x = x + noise

        if self.compress is not None:
            x = self.decompress(x)

        for layer in self.mlp[self.split_layer + 1:]:
            x = layer(x)

        return x


    def forward_for_jacobian(self, x):
        # If we do not have BN layer, this is simple
        return self.split_forward(x, return_act=True)
    
    def set_prune_p(self, p):
        self.prune_p = p

    def set_split_layer(self, layer):
        self.split_layer = layer

    def set_bn_training(self, training):
        pass

    def get_embs(self, repeat):
        # Ignore repeat: hack to be compatible with transformers
        return self.embs
