import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch
#from functorch import vmap, jvp
from torch.func import vmap, jvp
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
import random
import math
import time

from movielens.data_utils import MovieLensDataset
from attack_model import InversionNet

#distilbert_model_path = "./models/"

def get_dataset(dataset, standardize, data_portion=1.0, gaussian_mu=0.0, gaussian_sigma=1.0):
    if "cifar" in dataset:
        if standardize:
            data_mu = (0.4914, 0.4822, 0.4465)
            data_sigma = (0.2471, 0.2435, 0.2616)
            transform_train = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.Normalize(data_mu, data_sigma)])
            transform_test = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize(data_mu, data_sigma)])
        else:
            data_mu = None
            data_sigma = None
            transform_train = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip()])
            transform_test = transforms.Compose(
                    [transforms.ToTensor()])
        if dataset == 'cifar100':
            train_data = datasets.CIFAR100('./data/CIFAR100', train=True, transform=transform_train, download=True)
            test_data = datasets.CIFAR100('./data/CIFAR100', train=False, transform=transform_test, download=True)
        elif dataset == 'cifar10':
            train_data = datasets.CIFAR10('./data/CIFAR10', train=True, transform=transform_train, download=True)
            test_data = datasets.CIFAR10('./data/CIFAR10', train=False, transform=transform_test, download=True)
        else:
            raise NotImplementedError()

    elif dataset == "mnist":
        # Not implementing MNIST standardization.
        # MNIST is tested for proof of concept only on a toy model.
        data_mu = None
        data_sigma = None
        assert(not standardize)
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST('./data/MNIST', train=True, transform=transform, download=True)
        test_data = datasets.MNIST('./data/MNIST', train=False, transform=transform, download=True)

    elif dataset == "movielens-20":
        data_mu = None
        data_sigma = None
        train_data = MovieLensDataset(train=True, raw_path='./data/movielens/', processed_path='./data/movielens/processed/')
        test_data = MovieLensDataset(train=False, raw_path='./data/movielens/', processed_path='./data/movielens/processed/')

    elif dataset == "tinyimagenet":
        if standardize:
            data_mu = (0.4802, 0.4481, 0.3975)
            data_sigma = (0.2302, 0.2265, 0.2202)
            transform_train = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.RandomCrop(64, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.Normalize(data_mu, data_sigma)])
            transform_test = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize(data_mu, data_sigma)])
        else:
            data_mu = None
            data_sigma = None
            transform_train = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.RandomCrop(64, padding=4),
                     transforms.RandomHorizontalFlip()])
            transform_test = transforms.Compose(
                    [transforms.ToTensor()])
        train_data = datasets.ImageFolder('./data/tiny-imagenet-200/train', transform=transform_train)
        test_data = datasets.ImageFolder('./data/tiny-imagenet-200/val', transform=transform_test)

    elif dataset == "synthetic_gaussian":
        class RandomGaussianDataset(Dataset):
            def __init__(self, dim=784, mu=0.0, sigma=1.0, total_size=100):
                self.mu = torch.ones([dim], dtype=torch.float) * mu
                self.sigma = torch.ones([dim], dtype=torch.float) * sigma
                self.len = total_size
            def __len__(self):
                return self.len
            def __getitem__(self, idx):
                data = torch.normal(self.mu, self.sigma)
                return data, 0

        data_mu = None
        data_sigma = None
        train_data = None
        test_data = RandomGaussianDataset(mu=gaussian_mu, sigma=gaussian_sigma)

    elif dataset == "glue-sst2":
        from datasets import load_dataset
        dataset = load_dataset("glue", "sst2")
        train_data = None
        test_data = dataset["validation"]
        data_mu = None
        data_sigma = None

    if data_portion < 1.0:
        # Only use the subset of the train_data for training
        # Used for training the encoder for private training use-case.
        print(f"Before subsampling, {len(train_data)}")
        train_data = torch.utils.data.Subset(train_data, range(int(len(train_data) * data_portion)))
        print(f"After subsampling, {len(train_data)}")

    return train_data, test_data, data_mu, data_sigma


def get_attack_model(encoder_model, dataset, split_layer, last_activation, device):
    # Currently only implemented against resnet18
    assert(encoder_model == "resnet18")
    # InversionNet worked the best.
    if dataset == 'cifar10':
        if split_layer <= 2:
            inet = InversionNet(in_c=64, upconv_channels=[(128, 'same'), (3, 'same')], last_activation=last_activation)
        elif split_layer == 7:
            inet = InversionNet(last_activation=last_activation)
        elif split_layer == 9:
            inet = InversionNet(in_c=256, upconv_channels=[(256, 'up'), (256, 'up'), (3, 'up')], last_activation=last_activation)
    elif dataset == 'mnist':
        if split_layer == 0:
            inet = InversionNet(in_c=16, upconv_channels=[(32, 'same'), (1, 'same')], last_activation=last_activation)

    inet = inet.to(device)

    return inet


def get_model(model, dataset, split_layer, bottleneck_dim, activation, pooling, test_data, device, load_from_file=False, model_file=""):
    tokenizer = None
    if 'cifar' in dataset:
        from PyTorch_CIFAR10.cifar10_models.resnet import resnet18
        if model == "resnet18":
            net = resnet18(pretrained=False, split_layer=split_layer, bottleneck_dim=bottleneck_dim, num_classes=10 if dataset == 'cifar10' else 100, activation=activation, pooling=pooling)
        else:
            raise NotImplementedError()

    elif dataset == "mnist":
        # MNIST is tested for proof of concept only on a toy model.
        from PyTorch_CIFAR10.cifar10_models.cnn import CNN
        if model == "cnn":
            net = CNN(split_layer=split_layer)
        else:
            raise NotImplementedError()
        
    elif dataset == "movielens-20":
        from movielens.models import NCF_MLP
        if model == "ncf-mlp":
            net = NCF_MLP(emb_size=[test_data.max_uid + 1, test_data.max_mid + 1], mlp_dims=[64, 32, 16], split_layer=split_layer, bottleneck_dim=bottleneck_dim)
        else:
            raise NotImplementedError()

    elif dataset == "tinyimagenet":
        from PyTorch_CIFAR10.cifar10_models.resnet import resnet18
        if model == "resnet18":
            net = resnet18(pretrained=False, split_layer=split_layer, bottleneck_dim=bottleneck_dim, num_classes=200, activation=activation, pooling=pooling)
        else:
            raise AssertionError()

    elif dataset == "synthetic_gaussian":
        from PyTorch_CIFAR10.cifar10_models.mlp import MLP
        if model == "mlp-10k-untrained":
            net = MLP(split_layer=split_layer, bottleneck_dim=bottleneck_dim, in_dim=784, out_channels=[10000, 512, 64])
        elif model == "mlp-1k-untrained":
            net = MLP(split_layer=split_layer, bottleneck_dim=bottleneck_dim, in_dim=784, out_channels=[1000, 512, 64])

    elif dataset == "glue-sst2":
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

        #if jacloss_alpha == 0.0 and train_lb == 0.0:
        #    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        #    net = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        #else:
        tokenizer = DistilBertTokenizer.from_pretrained(model_file, local_files_only=True)
        net = DistilBertForSequenceClassification.from_pretrained(model_file, local_files_only=True)
        net.set_split_layer(split_layer)

    if load_from_file:
        print("Loading pretrained model...")
        if dataset == "glue-sst2":
            pass
        else:
            if device == "cpu":
                pretrained_dict = torch.load(model_file, map_location=torch.device('cpu'))
                #net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
            else:
                pretrained_dict = torch.load(model_file)
                #net.load_state_dict(torch.load(model_file))
            #print(pretrained_dict)
            if "cifar100" in model_file or "tinyimagenet" in model_file:
                assert(dataset == "cifar10")
                # Don't load the last layer for split learning, if the last layer does not match.
                # This is a bit hacky for now.
                init_dict = net.state_dict()
                pretrained_dict = {k: pretrained_dict[k] if k not in ["fc.weight", "fc.bias"] else init_dict[k] for k in pretrained_dict}
            net.load_state_dict(pretrained_dict)
    else:
        assert(dataset != "glue-sst2")
        print("Using untrained model")

    net = net.to(device)

    return net, tokenizer


def get_attack_model_name(model_path, model_file, dataset, split_layer, target_lb, input_noise, last_activation):
    suffix = f'_{last_activation}'
    if input_noise > 0.0:
        suffix += f'_input_noise{input_noise}'
    return f'{model_path}/attacker_{model_file.split("/")[-1]}_{dataset}_l{split_layer}_lb{target_lb}{suffix}.pt'

def get_model_name(model_path, model, dataset, split_layer, bottleneck_dim, jacloss_alpha, train_lb, standardize, input_noise, data_portion, activation, pooling, train_bs, train_seed):
    if model == "distilbert":
        model_file = f"{model_path}/{model}-{split_layer}-{jacloss_alpha}-{train_lb}-{train_seed}"
    else:
        suffix = ""
        if jacloss_alpha > 0.0:
            suffix += f"_jacloss{jacloss_alpha}"
        if train_lb > 0.0:
            suffix += f"_train_lb{train_lb}"
        if input_noise > 0.0:
            suffix += f"_input_noise{input_noise}"
        if data_portion < 1.0:
            suffix += f"_data_portion{data_portion}"
        suffix += f"_act_{activation}_pool_{pooling}_bs{train_bs}_seed{train_seed}"
        model_file = f"{model_path}/{model}_{dataset}_l{split_layer if (bottleneck_dim > 0 or jacloss_alpha > 0.0 or train_lb > 0) else -1}_b{bottleneck_dim}_standardize-{standardize}{suffix}.pt"

    return model_file

def get_optimizer(net, optim, lr, weight_decay=5e-4, nesterov=False):
    if optim == "adagrad":
        optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
    elif optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=nesterov)
    else:
        raise AssertionError()
    
    return optimizer


def de_normalize(img, mu, sigma):
    transform = transforms.Normalize([-m/s for m, s in zip(mu, sigma)], [1.0 / s for s in sigma])
    img = transform(img).unsqueeze(0)
    img[img > 1.0] = 1.0
    img[img < 0.0] = 0.0
    return img


def save_image(img, mu, sigma, name, path="./imgs/"):
    if mu is not None:
        img = de_normalize(img, mu, sigma)
    if not os.path.exists(path):
        os.makedirs(path)
    torchvision.utils.save_image(img, f"{path}/{name}")


def save_image_wrapper(x, res, prefix, num_save, path, data_mu=None, data_sigma=None):
    save_image(x, data_mu, data_sigma, f"{prefix}-ref.png", path=path)
    save_image(res, data_mu, data_sigma, f"{prefix}-recon.png", path=path)

    return num_save + 1


def calc_tr(net, x, device, subsample=-1, jvp_parallelism=1):
    def jvp_func(x, tgt):
        return jvp(net.forward_first, (x,), (tgt,))

    d = x[0].flatten().shape[0]
    tr = torch.zeros(x.shape[0], dtype=x.dtype).to(device)
    #print(f'd: {d}, {x.shape}')

    # Randomly subsample pixels for faster execution
    if subsample > 0:
        samples = random.sample(range(d), min(d, subsample))
    else:
        samples = range(d)

    #print(x.shape, d, samples)
    for j in range(math.ceil(len(samples) / jvp_parallelism)):
        tgts = []
        for k in samples[j*jvp_parallelism:(j+1)*jvp_parallelism]:
            tgt = torch.zeros_like(x).reshape(x.shape[0], -1)
            tgt[:, k] = 1.
            tgt = tgt.reshape(x.shape)
            tgts.append(tgt)
        tgts = torch.stack(tgts)
        def helper(tgt):
            vals, grad = vmap(jvp_func, randomness='same')(x, tgt)
            #print('grad shape: ', grad.shape)
            return torch.sum(grad * grad, dim=tuple(range(1, len(grad.shape)))), vals
        trs, vals = vmap(helper, randomness='same')(tgts) # randomness for randomness control of dropout
        # vals are stacked results that are repeated by d (should be all the same)

        tr += trs.sum(dim=0)

    # Scale if subsampled
    if subsample > 0:
        tr *= d / len(samples)
    return tr, vals[0].squeeze(1)  # squeeze removes one dimension jvp puts


# This can be merged with above, but separating for cleaner code
def calc_tr_transformer(net, inputs, inputs_embeds, device, subsample=-1, jvp_parallelism=1, tokenizer=None):
    def jvp_func(inputs_embeds, attention_mask, tgt):
        def forward_first_helper(x):
            return net.forward_first(x, attention_mask)

        return jvp(forward_first_helper, (inputs_embeds,), (tgt,))

    # This is a bit dirty, but this handles transformer models.
    #input_ids = x['input_ids'][0]
    mask_lens = inputs['attention_mask'].sum(dim=1)
    x = inputs_embeds
    d = x[0].flatten().shape[0]
    tr = torch.zeros(x.shape[0], dtype=x.dtype).to(device)
    #print(f'd: {d}, {x.shape}')

    # Randomly subsample pixels for faster execution
    if subsample > 0:
        # TODO: We should not subsample from an invalid token. However, we
        # also should subsample the same pos for all batches (when their seq_length are all different).
        # This causes a dillemma. Thus, let's try subsampling from the intersection,
        # which is incorrect, but may... work.
        # TODO: This really needs fix because we are training only the first few words more...
        #samples = random.sample(range(min(mask_lens) * x.shape[-1]), min(min(mask_lens) * x.shape[-1], subsample))
        samples = []
        for mask_len in mask_lens:
            mask_len = mask_len.item()
            samples.append(random.sample(range(mask_len * x.shape[-1]), min(mask_len * x.shape[-1], subsample)))
    else:
        # No need to run for entire inputs, because we only care about valid input
        samples = [range(min(d, max(mask_lens) * x.shape[-1]))]

    for j in range(math.ceil(len(samples[0]) / jvp_parallelism)):
        tgts = []
        if len(samples) == 1:
            for k in samples[0][j*jvp_parallelism:(j+1)*jvp_parallelism]:
                tgt = torch.zeros_like(x).reshape(x.shape[0], -1)
                tgt[:, k] = 1.
                tgt = tgt.reshape(x.shape)
                tgts.append(tgt)
        else:
            for k in range(j*jvp_parallelism, (j+1)*jvp_parallelism):
                tgt = torch.zeros_like(x).reshape(x.shape[0], -1)
                for b, sample in enumerate(samples):
                    tgt[b, sample[k]] = 1.
                tgt = tgt.reshape(x.shape)
                tgts.append(tgt)
        tgts = torch.stack(tgts)
        def helper(tgt):
            vals, grad = vmap(jvp_func, randomness='same')(x, inputs['attention_mask'], tgt)
            #print('grad shape: ', grad.shape)
            return torch.sum(grad * grad, dim=tuple(range(1, len(grad.shape)))), vals
        trs, vals = vmap(helper, randomness='same')(tgts) # randomness for randomness control of dropout
        # vals are stacked results that are repeated by d (should be all the same)

        if tokenizer is not None:
            z = vals[0].squeeze(1)
            z_inner = torch.bmm(z.view(z.shape[0], 1, -1), z.view(z.shape[0], -1, 1)).flatten()
            print(tokenizer.decode(inputs['input_ids'][0][j]).replace(' ', ''), sum(trs) / z_inner)
        #print(trs.detach().clone(), trs.shape)
        #print(vals.shape, vals[0].squeeze().shape)
        #print(vals[0][0].squeeze()[:30])
        tr += trs.sum(dim=0)
    #print(tr.shape, vals.shape)
    # Scale if subsampled
    if subsample > 0:
        # TODO: This is not exact
        tr *= (min(mask_lens) * x.shape[-1]) / len(samples)
    return tr, vals[0].squeeze(1)  # squeeze removes one dimension jvp puts


def calc_mean_tr(net, loader, calc_tr_fn, device, jvp_parallelism=100, subsample=-1, emb_func=None):
    net.eval()

    tr_sum = 0
    cnt = 0
    with torch.no_grad():
        for i, inputs in enumerate(loader):
            if calc_tr_fn == calc_tr_transformer:
                x = inputs
                y = inputs['labels']
            else:
                x = inputs[0]
                y = inputs[1]

            if isinstance(x, list):
                x = [t.to(device) for t in x]
            elif isinstance(x, dict):
                x = {k: v if k == 'sentence' else torch.tensor(v).to(device) for k, v in x.items()}
            else:
                x = x.to(device)
            if emb_func is not None:
                if isinstance(x, dict):
                    inputs_embeds = emb_func(x['input_ids'])
                else:
                    x = emb_func(x)
            y = y.to(device)

            if isinstance(x, dict):
                d = inputs_embeds[0].flatten().shape[0]
            else:
                d = x[0].flatten().shape[0]
            prev_t = time.time()
            if calc_tr_fn == calc_tr_transformer:
                tr, _ = calc_tr_fn(net, x, inputs_embeds, device, jvp_parallelism=jvp_parallelism, subsample=subsample)
            else:
                tr, _ = calc_tr_fn(net, x, device, jvp_parallelism=jvp_parallelism, subsample=subsample)
            print(i, len(loader), time.time() - prev_t)
            tr_sum += tr.sum().detach().item()
            cnt += y.shape[0]

    return tr_sum / cnt, d
