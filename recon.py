import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from util import save_image_wrapper

# TV Loss
### This code is directly copied from:
### https://github.com/jxgu1016/Total_Variation_Loss.pytorch/blob/master/TVLoss.py
def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow(x[:,:,1:,:] - x[:,:,:h_x-1,:], 2).sum()
    w_tv = torch.pow(x[:,:,:,1:] - x[:,:,:,:w_x-1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size

def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]
### Copied code ends here

num_recon = 0
num_save = 0

def recon_wrapper(net, x, act, sigma, recon_params, dataset, recon_name, reconstruct_path, data_mu, data_sigma, num_total_save, num_total_recon, device, inet, tokenizer):
    global num_save
    global num_recon

    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    lpips = lpips.to(device)

    # Recon each samples
    bs = 1 if isinstance(x, dict) else x.shape[0]
    for j in range(bs):
        if sigma is not None and len(sigma.shape) > 0.0:
            s = sigma[j]
        else:
            s = sigma

        print(f"Using sigma {s} for sample {j}")
        if recon_params["method"] in ["tv", "gaussian_map"]:
            res = recon_white_box(net, s, x if isinstance(x, dict) else x[j].unsqueeze(0), act if isinstance(x, dict) else act[j].unsqueeze(0), device, recon_params)
        elif recon_params["method"] == 'cnn':
            res = recon_black_box(inet, s, act[j].unsqueeze(0))

        # TODO: We really need to clean this post-processing up...
        if dataset in ["mnist", "cifar10"]:
            if num_save < num_total_save:
                num_save = save_image_wrapper(x[j], res[0], f"{dataset}{j}-{recon_name}", num_save, reconstruct_path, data_mu, data_sigma)
            try:
                lp = lpips(res, x[j].unsqueeze(0))
            except:
                lp = 0

            print(f"Recon {num_recon} MSE {torch.mean((res - x[j].unsqueeze(0)) ** 2)} PSNR {peak_signal_noise_ratio(res, x[j].unsqueeze(0))} SSIM {structural_similarity_index_measure(res, x[j].unsqueeze(0))}, LPIPS {lp}")

        elif dataset == "movielens-20":
            # Try finding the closest embedding.
            x_ref = x[j].flatten()
            x_recon = res[0].flatten()
            top1 = []
            top5 = []
            mses = []
            for i, emb in enumerate(net.embs):
                emb_size = emb.weight.shape[-1]
                max_idx = torch.topk(emb.weight @ x_recon[i * emb_size: (i + 1) * emb_size], 5).indices
                top1.append(torch.equal(emb.weight[max_idx[0]], (x_ref[i * emb_size: (i + 1) * emb_size])))
                top5.append(False)
                for idx in max_idx:
                    top5[-1] += torch.equal(emb.weight[idx], (x_ref[i * emb_size: (i + 1) * emb_size]))
                mses.append(torch.mean((x_recon[i * emb_size : (i + 1) * emb_size] - x_ref[i * emb_size : (i + 1) * emb_size]) ** 2))
            print(f"MSE {mses} {top1} {top5}")
            print(torch.mean(torch.tensor(top1, dtype=torch.float)), torch.mean(torch.tensor(top5, dtype=torch.float)))

        elif dataset == "glue-sst2":
            x_ref = x['inputs_embeds'][j].flatten()
            sentence_len = x['attention_mask'][j].shape[-1]
            input_ids = x['input_ids'][j].flatten()
            x_recon = res.flatten()
            
            top1 = []
            top5 = []
            mses = []

            recon_ids = []
            for i in range(sentence_len):
                emb_size = net.distilbert.embeddings.word_embeddings.weight.shape[-1]
                emb_normed = net.distilbert.embeddings.LayerNorm(net.distilbert.embeddings.word_embeddings.weight + net.distilbert.embeddings.position_embeddings(torch.tensor(i).to(device)))

                max_idx = torch.topk(emb_normed @ x_recon[i * emb_size: (i + 1) * emb_size], 5).indices

                top1.append(max_idx[0] == input_ids[i])
                top5.append(False)
                for idx in max_idx:
                    top5[-1] += idx == input_ids[i]
                mses.append(torch.mean((x_recon[i * emb_size : (i + 1) * emb_size] - x_ref[i * emb_size : (i + 1) * emb_size]) ** 2))
                recon_ids.append(max_idx[0])
            
            print('Original string: ', tokenizer.decode(x['input_ids'][0]))
            print('Reconstructed string: ', tokenizer.decode(recon_ids))
            print("len T1 T5 MSE:", len(top1), torch.mean(torch.tensor(top1, dtype=torch.float)), torch.mean(torch.tensor(top5, dtype=torch.float)), torch.mean(torch.tensor(mses, dtype=torch.float)))

        num_recon += 1
        if num_recon >= num_total_recon:
            return True

    return False


def recon_white_box(net, sigma, x_ref, a_ref, device, recon_params):
    # Only works if we run image-by-image
    lr = recon_params['lr']
    stop_criterion = recon_params['stop_criterion']
    niters = recon_params['niters']
    method = recon_params['method']

    a_ref = a_ref.detach()
    if sigma is not None:
        a_ref += torch.normal(torch.zeros_like(a_ref), sigma)

    # TODO: This means the model is transformer.
    # It is a bit of a dirty hack..
    if isinstance(x_ref, dict):
        assert(x_ref['inputs_embeds'].shape[0] == 1)
        x_recon = torch.normal(torch.zeros_like(x_ref['inputs_embeds']), 1) * 0.1
        attn_mask = x_ref['attention_mask']
        x_recon = x_recon.squeeze(0)
    else:
        assert(x_ref.shape[0] == 1)
        x_recon = torch.normal(torch.zeros_like(x_ref), 1) * 0.1

    x_recon.requires_grad = True
    x_recon = x_recon.to(device)
    optimizer = optim.Adam(params=[x_recon], lr=lr)

    prev_window_loss = None
    window_loss = 0
    for i in range(niters):
        if isinstance(x_ref, dict):
            a_recon = net.forward_first(x_recon, attn_mask)
        else:
            a_recon = net.forward_first(x_recon, for_jacobian=False)
        mse_loss = ((a_ref - a_recon) ** 2).mean() / (a_ref ** 2).mean()
        if method == 'tv':
            if recon_params['tv_lmbda'] > 0.0:
                prior_loss = recon_params['tv_lmbda'] * TV(x_recon)
            else:
                prior_loss = torch.zeros_like(mse_loss)
        elif method == 'gaussian_map':
            prior_loss = recon_params['gaussian_lmbda'] * torch.mean(((x_recon - recon_params['gaussian_mu']) / recon_params['gaussian_sigma']) ** 2)
        else:
            assert(False)
        loss = mse_loss + prior_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        window_loss += loss.detach()

        if i != 0 and i % 100 == 0:
            window_loss /= 100
            if isinstance(x_ref, dict):
                mse = torch.mean((x_ref['inputs_embeds'] - x_recon) ** 2)
            else:
                mse = torch.mean((x_ref - x_recon) ** 2)
            print(f"{i} loss {window_loss} {prior_loss} prev_loss {prev_window_loss}, res loss {mse}")
            #print(x_ref, x_recon)
            if prev_window_loss is not None and abs(prev_window_loss - window_loss) / prev_window_loss < stop_criterion:
                print(f"Break at {i}")
                break
            prev_window_loss = window_loss
            window_loss = 0

    return x_recon


def recon_black_box(inet, sigma, a_ref):
    a_ref = a_ref.detach()
    if sigma is not None:
        a_ref = a_ref + torch.normal(torch.zeros_like(a_ref), sigma)

    return inet(a_ref)
