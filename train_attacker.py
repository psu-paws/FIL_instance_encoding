from attack_model import InversionNet, Hourglass
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import random
from util import calc_tr, calc_mean_tr
from util import calc_tr, calc_mean_tr, get_dataset, get_model, get_model_name, get_attack_model, get_attack_model_name, get_optimizer

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="./models")

    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--split-layer', type=int, default=7)
    parser.add_argument('--bottleneck-dim', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--standardize', action="store_true", default=False)
    parser.add_argument('--target-lb', type=float, default=1e-5)
    parser.add_argument('--jvp-parallelism', type=int, default=1000)
    parser.add_argument('--input-noise', type=float, default=0.0)
    parser.add_argument('--last-activation', type=str, default='None')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--encoder-model', type=str, default='resnet18')
    parser.add_argument('--encoder-file', type=str, default='')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--pooling', type=str, default='max')

    args = parser.parse_args()

    return args

def train_attacker(args):
    print(args, flush=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    attack_model_file = get_attack_model_name(args.model_path, args.encoder_file, args.dataset, args.split_layer, args.target_lb, args.input_noise, args.last_activation)

    train_data, test_data, *_ = get_dataset(args.dataset, args.standardize)

    net, _ = get_model(args.encoder_model, args.dataset, args.split_layer, args.bottleneck_dim, args.activation, args.pooling, test_data, device, load_from_file=True, model_file=args.encoder_file)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=0)

    # First part of this model is used as an encoder
    net = net.to(device)
    net.eval()
    print(net)

    tr_sum = 0
    cnt = 0
    if args.target_lb > 0:
        tr_mean, d = calc_mean_tr(net, test_loader, calc_tr, device, args.jvp_parallelism)
        sigma = torch.sqrt(torch.tensor(args.target_lb * tr_mean / d)).to(device)
        print(f'tr mean: {tr_mean}, sigma {sigma}')
    else:
        sigma = 0

    inet = get_attack_model(args.encoder_model, args.dataset, args.split_layer, args.last_activation, device)
    print(inet)

    inet.train()

    loss_func = nn.MSELoss()
    optimizer = get_optimizer(inet, "adam", args.lr)

    for epoch in range(args.epochs):
        inet.train()
        total_loss = 0
        n = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            if args.input_noise > 0.0:
                # Noisy input (randomized smoothing)
                x = x + torch.normal(torch.zeros_like(x), args.input_noise)
            # Calculate the encoding
            a = net.forward_first(x, for_jacobian=False)
            if args.target_lb > 0:
                # Add noise during training
                a = a + torch.normal(torch.zeros_like(a), sigma)
            # Try reconstructing the original input
            x_new = inet(a)
            loss = loss_func(x_new, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * y.shape[0]
            n += y.shape[0]
            if (i + 1) % args.print_every == 0:
                print(f"Epoch {epoch} iter {i} loss {total_loss / n}", flush=True)
                total_loss = 0
                n = 0

        # Eval after every epoch
        inet.eval()
        total_loss = 0
        n = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                if args.input_noise > 0.0:
                    # Noisy input (randomized smoothing)
                    x += torch.normal(torch.zeros_like(x), args.input_noise)
                a = net.forward_first(x, for_jacobian=False)
                if args.target_lb > 0:
                    a = a + torch.normal(torch.zeros_like(a), sigma)
                x_new = inet(a)
                # Recon orig image
                loss = loss_func(x_new, x)
                total_loss += loss.detach().item() * y.shape[0]
                n += y.shape[0]
            print(f"Epoch {epoch} test loss {total_loss / n}", flush=True)

    torch.save(inet.state_dict(), attack_model_file)

if __name__ == "__main__":
    args = get_args()
    train_attacker(args)
