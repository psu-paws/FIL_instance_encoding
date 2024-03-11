import time
import random
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch import nn

from recon import recon_wrapper
from util import calc_tr, calc_tr_transformer, calc_mean_tr, get_dataset, get_model, get_model_name, get_optimizer, get_attack_model_name, get_attack_model
from attack_model import InversionNet

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="./models")
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--train-bs', type=int, default=128)
    parser.add_argument('--test-bs', type=int, default=128)
    parser.add_argument('--split-layer', type=int, default=7)
    parser.add_argument('--bottleneck-dim', type=int, default=-1)
    parser.add_argument('--num-recon', type=int, default=10)
    parser.add_argument('--num-save', type=int, default=10)
    parser.add_argument('--target-lb', type=float, default=1e-5)
    parser.add_argument('--recon-lr', type=float, default=1e-1) # 1.0
    parser.add_argument('--tv-lmbda', type=float, default=5e-2) # 1.0
    parser.add_argument('--gaussian-lmbda', type=float, default=5e-2) # 1.0
    parser.add_argument('--gaussian-mu', type=float, default=0.0) # 1.0
    parser.add_argument('--gaussian-sigma', type=float, default=1.0) # 1.0
    parser.add_argument('--recon-iter', type=int, default=10000)
    parser.add_argument('--recon-stop', type=float, default=1e-5)
    parser.add_argument('--standardize', action="store_true", default=False)
    parser.add_argument('--reconstruct', action="store_true", default=False)
    parser.add_argument('--reconstruct-method', type=str, default="tv") # tv, dip
    parser.add_argument('--reconstruct-path', type=str, default="./imgs")
    parser.add_argument('--subsample-FIL', type=int, default=-1)
    parser.add_argument('--jacloss-alpha', type=float, default=0.0) # With CIFAR-10, 0.1 seems to improve the accuracy.
    parser.add_argument('--jvp-parallelism', type=int, default=1) # Increasing this as much as possible without blowing up the memory makes J calc fast
    parser.add_argument('--train-lb', type=float, default=0.0) # Add a train-time noise
    parser.add_argument('--input-noise', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--last-activation', type=str, default='None')
    parser.add_argument('--pooling', type=str, default='avg')
    parser.add_argument('--train-seed', type=int, default=123)
    parser.add_argument('--test-seed', type=int, default=0)
    parser.add_argument('--load-from-file', action='store_true', default=False)

    args = parser.parse_args()

    return args

def run(args):
    print(args)
    torch.manual_seed(args.test_seed)
    random.seed(args.test_seed)
    np.random.seed(args.test_seed)

    if torch.cuda.is_available():
        device = f"cuda:0"
    else:
        device = "cpu"

    model_file = get_model_name(args.model_path, args.model, args.dataset, args.split_layer, args.bottleneck_dim, args.jacloss_alpha, args.train_lb, args.standardize, args.input_noise, 1.0, args.activation, args.pooling, args.train_bs, args.train_seed)
    print(model_file)

    _, test_data, data_mu, data_sigma = get_dataset(args.dataset, args.standardize, gaussian_mu=args.gaussian_mu, gaussian_sigma=args.gaussian_sigma)
    net, tokenizer = get_model(args.model, args.dataset, args.split_layer, args.bottleneck_dim, args.activation, args.pooling, test_data, device, args.load_from_file, model_file)
    
    if tokenizer is not None:
        if args.test_bs > 1:
            raise AssertionError("Currently, I haven't implemented bs > 1 for transformers")
        def encode(batch):
            features = tokenizer(batch["sentence"], return_tensors="pt", padding=True, truncation=True)
            # Change label to labels
            features["labels"] = batch["label"]
            return features
        test_data = test_data.map(encode, remove_columns=['label'])

    if args.dataset in ["cifar10", "cifar100", "mnist"]:
        loss_func = nn.CrossEntropyLoss()
        metrics = ["loss", "accuracy", "accuracy5"]
    elif args.dataset == "synthetic_gaussian":
        loss_func = nn.CrossEntropyLoss()
        metrics = ["loss", "accuracy", "accuracy5"]
    elif args.dataset == "movielens-20":
        loss_func = nn.BCELoss()
        metrics = ["loss", "auc"]
    elif args.dataset == "glue-sst2":
        metrics = ["loss", "accuracy"]

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False, num_workers=0)

    net.eval()
    print(net)

    if args.target_lb > 0.0:
        tr_mean, d = calc_mean_tr(net, test_loader, calc_tr_transformer if args.dataset == "glue-sst2" else calc_tr, device, args.jvp_parallelism, emb_func=net.forward_embs if args.dataset in ["movielens-20", "glue-sst2"] else None)
        sigma = torch.sqrt(torch.tensor(args.target_lb * tr_mean / d)).to(device)
        print(f'tr mean: {tr_mean}, sigma {sigma}')
    else:
        sigma = None

    recon_params = {'method': args.reconstruct_method,
            'tv_lmbda': args.tv_lmbda,
            'gaussian_lmbda': args.gaussian_lmbda,
            'gaussian_mu': args.gaussian_mu,
            'gaussian_sigma': args.gaussian_sigma,
            'niters': args.recon_iter,
            'stop_criterion': args.recon_stop,
            'lr': args.recon_lr
            }

    n = 0
    total_t = 0
    total_loss = 0
    total_acc = 0
    total_acc5 = 0
    total_auc = 0

    y_preds = torch.tensor([]).to(device)
    ans = torch.tensor([]).to(device)
    print(f"{len(test_loader)}")
    for i, inputs in enumerate(test_loader):
        if args.dataset == "glue-sst2":
            x = inputs
            y = inputs['labels']
        else:
            x = inputs[0]
            y = inputs[1]

        if args.dataset == "movielens-20":
            x = [t.to(device) for t in x]
        elif args.dataset == "glue-sst2":
            x = {k: v if k == 'sentence' else torch.tensor(v).to(device) for k, v in x.items()}
        else:
            x = x.to(device)
        y = y.to(device)

        if args.dataset == "movielens-20":
            x = net.forward_embs(x)
        elif args.dataset == "glue-sst2":
            inputs_embeds = net.forward_embs(x['input_ids'])
            x["inputs_embeds"] = inputs_embeds.detach().clone()

        if args.input_noise > 0.0:
            x = x + torch.normal(torch.zeros_like(x), args.input_noise)

        # Run noisy inference
        with torch.no_grad():
            # dFIL-based noise addition
            if args.dataset == "glue-sst2":
                act = net.forward_first(inputs_embeds, x['attention_mask'], for_jacobian=False)
                # We only cut between the transformer blocks,
                # so do not need to consider cloning (it is mainly due to inplace operators)
                y_pred = net.forward_second(x, act, sigma=sigma)
                loss = y_pred.loss
                y_pred = y_pred.logits
            else:
                act = net.forward_first(x, for_jacobian=False)
                print(sigma)
                y_pred = net.forward_second(act.detach().clone(), sigma=sigma)
                loss = loss_func(y_pred, y)

            total_loss += loss.detach().item() * y.shape[0]

            # Calculating statistics for each batch
            acc = (torch.max(y_pred, 1)[1] == y).sum() if "accuracy" in metrics else 0
            acc5 = 0
            if "accuracy5" in metrics:
                top5 = torch.topk(y_pred, 5)[1]
                for j in range(5):
                    acc5 += (top5[:, j] == y).sum()

            total_acc += acc
            total_acc5 += acc5

            # AUC for this batch for printing
            try:
                auc = roc_auc_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy()) if "auc" in metrics else 0
            except:
                auc = 0

            y_preds = torch.cat([y_preds, y_pred.squeeze()])
            if "auc" in metrics:
                ans = torch.cat([ans, y.squeeze()])
            n += y.shape[0]

            print(f"loss {loss}, acc {acc / y.shape[0]}, acc5 {acc5 / y.shape[0]}, auc {auc}")

        # Run reconstruction
        if args.reconstruct:
            # Load attack model
            # Attack model hyperparam only found for certain setups.
            if args.reconstruct_method == "cnn":
                imodel_file = get_attack_model_name(args.model_path, model_file, args.dataset, args.split_layer, args.target_lb, args.input_noise, args.last_activation)
                inet = get_attack_model(args.model, args.dataset, args.split_layer, args.last_activation, device)

                print(imodel_file)
                inet.load_state_dict(torch.load(imodel_file))
                print(inet)
                inet.eval()
                inet = inet.to(device)
            else:
                inet = None

            recon_finished = recon_wrapper(net, x, act, sigma, recon_params, args.dataset,
                f"{args.model}-{args.reconstruct_method}-layer{args.split_layer}-bott{args.bottleneck_dim}-target{args.target_lb}.png",
                args.reconstruct_path, data_mu, data_sigma, args.num_save, args.num_recon, device, inet, tokenizer)
        
            if recon_finished:
                break

    metric_str = f"Eval "
    if "loss" in metrics:
        metric_str += f", loss {total_loss / n}"
    if "accuracy" in metrics:
        metric_str += f", acc {total_acc / n}, acc5 {total_acc5 / n}"
    if "auc" in metrics:
        metric_str += f", auc {round(roc_auc_score(ans.cpu().detach().numpy(), y_preds.cpu().detach().numpy()), 4)}"
    print(metric_str)

if __name__ == '__main__':
    args = get_args()
    run(args)
