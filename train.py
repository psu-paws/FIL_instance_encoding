import os
import time
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import argparse

import torch
from torch import nn
from torch.utils.data import random_split

from PyTorch_CIFAR10.schduler import WarmupCosineLR
from util import calc_tr, calc_mean_tr, get_dataset, get_model, get_model_name, get_optimizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--model-path', type=str, default="./models")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--print-every', type=int, default=99999999)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--split-layer', type=int, default=-1)
    parser.add_argument('--bottleneck-dim', type=int, default=-1)
    parser.add_argument('--standardize', action="store_true", default=False)
    parser.add_argument('--metrics', type=str, default="loss,accuracy")
    parser.add_argument('--test-fil', action="store_true", default=False)
    parser.add_argument('--jacloss-alpha', type=float, default=0.0) # With CIFAR-10, 0.1 seems to improve the accuracy.
    parser.add_argument('--jvp-parallelism', type=int, default=1)
    parser.add_argument('--train-lb', type=float, default=0.0) # Add a train-time noise
    parser.add_argument('--target-lb', type=float, default=1.0) # Noise for privacy (inference or split_learning).
    parser.add_argument('--optimizer', type=str, default="sgd") # Add a train-time noise
    parser.add_argument('--scheduler', type=str, default="warmupcosine") # Add a train-time noise
    parser.add_argument('--input-noise', type=float, default=0.0)
    parser.add_argument('--data-portion', type=float, default=1.0) # Portion of data to use
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--pooling', type=str, default='max')
    parser.add_argument('--seed', type=int, default=0) # Portion of data to use
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--split-learning', action='store_true', default=False)
    parser.add_argument('--encoder-file', type=str, default='')

    args = parser.parse_args()

    return args

def train(args):
    print(args)
    # transformer training should be done with run_glue.py
    assert(args.model != "distilbert")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    metrics = args.metrics.split(",")
    if torch.cuda.is_available():
        device = f"cuda:0"
    else:
        device = "cpu"

    train_data, test_data, *_ = get_dataset(args.dataset, args.standardize, args.data_portion)

    if args.validate:
        train_len = len(train_data)
        val_len = train_len // 10
        train_len -= val_len
        train_data, val_data = random_split(train_data, [train_len, val_len])
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=0)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=0)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=4)
    net, _ = get_model(args.model, args.dataset, args.split_layer, args.bottleneck_dim, args.activation, args.pooling, test_data, device, args.split_learning, args.encoder_file)

    optimizer = get_optimizer(net, args.optimizer, args.lr, args.weight_decay, args.nesterov)

    # For private training.
    # Note: This is not the so-called "split learning"
    # as encoder is not updated.
    if args.split_learning:
        # Currently only implemented for CIFAR10.
        assert(args.dataset == "cifar10")

        #net.create_new_fc(10)
        net.freeze_up_to_split()
        net = net.to(device)

        # Calculate the noise needed to added for the encodings.
        net.eval()
        tr_mean, d = calc_mean_tr(net, train_loader, calc_tr, device, args.jvp_parallelism, emb_func=net.forward_embs if args.dataset == "movielens-20" else None)
        sigma = torch.sqrt(torch.tensor(args.target_lb * tr_mean / d)).to(device)
        print(f'tr mean: {tr_mean}, sigma {sigma}')

        # We can generate more embeddings per input, each with more noise.
        # Here, we only implement creating only one.
        new_x = torch.tensor([]).to(device)
        new_y = torch.tensor([], dtype=torch.int64).to(device)
        if args.target_lb > 0:
            with torch.no_grad():
                for x, y in train_loader:
                    x = x.to(device)
                    y = y.to(device)
                    d = x[0].flatten().shape[0]
                    act = net.forward_first(x, for_jacobian=False)
                    act += torch.normal(torch.zeros_like(act), sigma)
                    new_x = torch.concat([new_x, act.detach().clone()], dim=0)
                    new_y = torch.concat([new_y, y.detach().clone()], dim=0)

            new_train_data = torch.utils.data.TensorDataset(new_x, new_y)
            train_loader = torch.utils.data.DataLoader(new_train_data, batch_size=args.bs, shuffle=True, num_workers=0)


    if args.dataset in ["cifar10", 'cifar100', "mnist", "tinyimagenet"]:
        total_steps = args.epochs * len(train_loader)
        loss_func = nn.CrossEntropyLoss()
    else:
        loss_func = nn.BCELoss()

    # TODO: Hardcode for now
    if args.scheduler == 'warmupcosine':
        scheduler = WarmupCosineLR(optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.02, epochs=args.epochs, steps_per_epoch=len(train_loader), div_factor=10, final_div_factor=10, pct_start=10/args.epochs)
    else:
        scheduler = None

    model_file = get_model_name(args.model_path, args.model, args.dataset, args.split_layer, args.bottleneck_dim, args.jacloss_alpha, args.train_lb, args.standardize, args.input_noise, args.data_portion, args.activation, args.pooling, args.bs, args.seed)
    print(model_file)

    print(net)
    print(len(train_loader))

    best_chkpt = args.encoder_file.split(".pt")[0] + f"lr{args.lr}_bs{args.bs}_weight_decay{args.weight_decay}_split_learning_best.pt"
    best_acc = 0.

    if not os.path.exists(model_file) or args.split_learning:
        for epoch in range(args.epochs):
            net.train()
            total_loss = 0
            total_acc = 0
            total_epoch_loss = 0
            total_epoch_acc = 0
            n = 0
            epoch_n = 0

            start_t = time.time()
            for i, (x, y) in enumerate(train_loader):
                if isinstance(x, list):
                    # This is for SST2
                    x = [t.to(device) for t in x]
                else:
                    x = x.to(device)
                y = y.to(device)

                if args.split_learning:
                    assert(args.dataset == "cifar10")
                    y_pred = net.forward_second(x)
                    loss = loss_func(y_pred, y)

                elif args.train_lb > 0.0:
                    if args.dataset == "movielens-20":
                        # We use the input embedding as an input for MovieLens
                        x = net.forward_embs(x)

                    if args.input_noise > 0.0:
                        # Add randomized smoothing noise.
                        x = x + torch.normal(torch.zeros_like(x), args.input_noise)

                    d = x[0].flatten().shape[0]

                    if args.jacloss_alpha > 0.0:
                        z = net.forward_first(x, for_jacobian=False)
                        net.set_bn_training(False)
                        tr, _ = calc_tr(net, x, device, jvp_parallelism=args.jvp_parallelism, subsample=10)
                        net.set_bn_training(True)
                        z_inner = torch.bmm(z.view(z.shape[0], 1, -1), z.view(z.shape[0], -1, 1)).flatten()
                        jac_loss = (tr / z_inner).mean()

                        sigma = torch.sqrt(args.train_lb * tr.detach().clone() / d)
                        y_pred = net.forward_second(z, sigma=sigma)
                        pred_loss = loss_func(y_pred, y)
                        loss = args.jacloss_alpha * jac_loss + pred_loss
                    else:
                        with torch.no_grad():
                            net.set_bn_training(False)
                            tr, _ = calc_tr(net, x, device, jvp_parallelism=args.jvp_parallelism, subsample=10)
                            net.set_bn_training(True)
                        sigma = torch.sqrt(args.train_lb * tr / d)

                        z = net.forward_first(x, for_jacobian=False)
                        y_pred = net.forward_second(z, sigma=sigma)
                        loss = loss_func(y_pred, y)

                elif args.jacloss_alpha > 0.0:
                    if args.dataset == "movielens-20":
                        x = net.forward_embs(x)
                    if args.input_noise > 0.0:
                        x = x + torch.normal(torch.zeros_like(x), args.input_noise)
                    z = net.forward_first(x, for_jacobian=False)
                    y_pred = net.forward_second(z)
                    pred_loss = loss_func(y_pred, y)
                    net.set_bn_training(False)
                    tr, _ = calc_tr(net, x, device, jvp_parallelism=args.jvp_parallelism, subsample=10)
                    net.set_bn_training(True)
                    z_inner = torch.bmm(z.view(z.shape[0], 1, -1), z.view(z.shape[0], -1, 1)).flatten()
                    jac_loss = (tr / z_inner).mean()
                    loss = args.jacloss_alpha * jac_loss + pred_loss
                else:
                    if args.input_noise > 0.0:
                        x = x + torch.normal(torch.zeros_like(x), args.input_noise)
                    y_pred = net(x)
                    loss = loss_func(y_pred, y)


                total_loss += loss.detach().item() * y.shape[0]
                if "accuracy" in metrics:
                    acc = (torch.max(y_pred, 1)[1] == y).sum()
                    total_acc += acc
                total_epoch_loss += loss.detach().item() * y.shape[0]
                if "accuracy" in metrics:
                    total_epoch_acc += acc
                n += y.shape[0]
                epoch_n += y.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % args.print_every == 0:
                    print(f"Epoch {epoch} iter {i} loss {total_loss / n}, acc {total_acc / n} elapsed {time.time() - start_t}")
                    total_loss = 0
                    total_acc = 0
                    n = 0
                    start_t = time.time()

                if scheduler is not None:
                    scheduler.step()

            print(f"Epoch {epoch}, loss {total_epoch_loss / epoch_n}, acc {total_epoch_acc / epoch_n}")

            if args.validate:
                if not args.split_learning:
                    raise AssertionError("Validation is only for split learning, because otherwise it is too slow.")
                net.eval()

                total_loss = 0
                total_acc = 0
                n = 0
                with torch.no_grad():
                    for i, (x, y) in enumerate(val_loader):
                        x = x.to(device)
                        y = y.to(device)
                        y_pred = net(x)
                        loss = loss_func(y_pred, y)
                        acc = (torch.max(y_pred, 1)[1] == y).sum()
                        total_acc += acc
                        total_loss += loss.detach().item() * y.shape[0]
                        n += y.shape[0]
                metric_str = f"Validate at epoch {epoch}"
                metric_str += f", loss {total_loss / n}"
                metric_str += f", acc {total_acc / n}"
                print(metric_str)

                if total_acc / n > best_acc:
                    best_acc = total_acc / n
                    torch.save(net.state_dict(), best_chkpt)

        if args.save_model and not args.validate:
            torch.save(net.state_dict(), model_file)
    else:
        print(f"Model {model_file} already exist")
        net.load_state_dict(torch.load(model_file))

    if args.validate:
        # Validation only for split learning
        assert(args.split_learning)
        if device == "cpu":
            net.load_state_dict(torch.load(best_chkpt, map_location=torch.device('cpu')))
        else:
            net.load_state_dict(torch.load(best_chkpt))

    if args.test_fil:
        # At the end, run noisy inference
        if args.train_lb == 0:
            target_lb = args.target_lb
        else:
            target_lb = args.train_lb

        split_layer = args.split_layer
        print(f"Layer {split_layer} target {target_lb}...")

        net.eval()

        if args.split_learning:
            tr_mean = None
            sigma = torch.tensor(0.).to(device)
        else:
            tr_mean, d = calc_mean_tr(net, test_loader, calc_tr, device, args.jvp_parallelism, emb_func=net.forward_embs if args.dataset == "movielens-20" else None)
            sigma = torch.sqrt(torch.tensor(target_lb * tr_mean / d)).to(device)
        print(f'Eval tr mean: {tr_mean}, sigma {sigma}')

        n = 0
        total_t = 0
        total_loss = 0
        total_acc = 0
        total_acc5 = 0
        total_auc = 0
        print(len(test_loader))
        if "auc" in metrics:
            y_preds = torch.tensor([]).to(device)
            ans = torch.tensor([]).to(device)
        for i, (x, y) in enumerate(test_loader):
            if isinstance(x, list):
                x = [t.to(device) for t in x]
            else:
                x = x.to(device)
            if args.dataset == "movielens-20":
                x = net.forward_embs(x)
            if args.input_noise > 0.0:
                x = x + torch.normal(torch.zeros_like(x), args.input_noise)
            y = y.to(device)
            start_t = time.time()
            with torch.no_grad():
                act = net.forward_first(x, for_jacobian=False)
                y_pred = net.forward_second(act, sigma=sigma)
                loss = loss_func(y_pred, y)
                total_loss += loss.detach().item() * y.shape[0]
                if "accuracy" in metrics:
                    acc = (torch.max(y_pred, 1)[1] == y).sum()
                    total_acc += acc
                    top5 = torch.topk(y_pred, 5)[1]
                    acc5 = 0
                    for j in range(5):
                        acc5 += (top5[:, j] == y).sum()
                    total_acc5 += acc5
                else:
                    acc = 0
                    acc5 = 0
                if "auc" in metrics:
                    y_preds = torch.cat([y_preds, y_pred.squeeze()])
                    ans = torch.cat([ans, y.squeeze()])
                n += y.shape[0]

                print(f"Average loss {total_loss / n}, acc {total_acc / n}, acc5 {total_acc5 / n}, auc {total_auc / n}")

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
    train(args)
