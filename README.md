# Bounding the Invertibility of Privacy-Preserving Instance Encoding Using Fisher Information
![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)
--------------------------------------------------------------------------------
Kiwan Maeng\*, Chuan Guo\*, Sanjay Kariyappa, and G. Edward Suh

NeurIPS 2023

## Install dependencies
```bat
pip install -r requirements.txt
```

## Setup
After cloning the `sliced_score_matching` submodule, we need to apply the following patch:
```bat
./apply_patch.sh
```

## Download datasets

### Download TinyImagenet
```bat
cd data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
python3 tinyimagenet_preprocess.py data/tiny-imagenet-200/val/
```

### Download MovieLens-20M
```bat
cd data
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip 
mv ml-20m movielens
```

## Brief explanation of the code
* `train.py` is for training models except for DistilBert. It is used for training models for case study 1 and pretraining/finetuning for case study 2.
* `run_glue.py` is for training DistilBert.
* `train_attacker.py` is for training DNN-based attackers.
* `run.py` is for evaluating split inference and running reconstruction attacks. 

## Sample executions
Hyperparameter not tuned to match the results from the paper.

### Reconstruct Gaussian (Figure 1)
```bat
python3 run.py --reconstruct --num-recon 100 --jvp-parallelism 100 --split-layer 0 --reconstruct-method gaussian_map --target-lb 1e-1 --model mlp-10k-untrained --dataset synthetic_gaussian --reconstruct --recon-lr 1e-3 --gaussian-lmbda 1e-1 --gaussian-sigma 0.05
```

### Reconstruct CIFAR10 (Figure 3; attack-ub)
```bat
python3 run.py --input-noise 0.25 --reconstruct --num-save 10 --num-recon 10 --jvp-parallelism 100 --split-layer 0 --reconstruct-method tv --target-lb 10.0 --dataset cifar10 --model resnet18
```

### Reconstruct CIFAR10 (Figure 5; attack-b) for split inference
```bat
python3 train.py --dataset cifar10 --model resnet18 --activation gelu --bs 128 --lr 0.1 --weight-decay 5e-4 --standardize --nesterov --test-fil --pooling avg --seed 123 --split-layer 7 --bottleneck-dim 8 --train-lb 1.0 --jvp-parallelism 100 --jacloss-alpha 0.0 --save-model
python3 train_attacker.py --dataset cifar10 --encoder-model resnet18 --activation gelu --lr 1e-4 --target-lb 1. --bs 64 --standardize --pooling avg --seed 123 --split-layer 7 --jvp-parallelism 100 --encoder-file models/resnet18_cifar10_l7_b-1_standardize-True_train_lb_1.0_act_gelu_pool_avg_bs128_seed123.pt
python3 run.py --reconstruct --num-save 10 --num-recon 10 --jvp-parallelism 100 --split-layer 7 --reconstruct-method cnn --target-lb 1.0  --standardize --activation gelu --train-lb 1. --pooling avg --load-from-file
```

### Reconstruct MovieLens-20M (Figure 6; left) for split inference
```bat
python3 train.py --dataset movielens-20 --model ncf-mlp --bs 128 --lr 0.1 --eval-every 1 --epochs 1 --metrics "loss,auc" --test-fil --jvp-parallelism 100 --split-layer 0 --scheduler "none" --train-lb 1. --seed 123 --print-every 1410 --optimizer adagrad --save-model --jacloss-alpha 0.01
python3 run.py --reconstruct --num-save 10 --num-recon 100 --jvp-parallelism 100 --split-layer 0 --reconstruct-method tv --tv-lmbda 0 --model ncf-mlp --dataset movielens-20 --pooling max --train-lb 1. --target-lb 1.0 --train-seed 123 --train-bs 128 --load-from-file --recon-iter 50000 --recon-lr 1e-2 
```

### Reconstruct and run GLUE-SST2 (Figure 6; right) for split inference
```bat
python3 run_glue.py --split-layer 2 --train-lb 10 --jacloss-alpha 0.01
python3 run.py --reconstruct --num-save 10 --num-recon 100 --jvp-parallelism 100 --split-layer 2 --reconstruct-method tv --tv-lmbda 0 --model distilbert --dataset glue-sst2 --train-lb 10.0 --target-lb 10.0 --train-seed 123 --load-from-file --recon-lr 1e-2 --jacloss-alpha 0.01 --test-bs 1
python3 run.py --dataset glue-sst2 --jvp-parallelism 768 --split-layer 2 --target-lb 10. --model distilbert --train-lb 10. --test-bs 1 --train-seed 123 --load-from-file --jacloss-alpha 0.01
```

### Split training (CIFAR100 -> CIFAR10)
```bat
python3 train.py --dataset cifar100 --model resnet18 --activation gelu --bs 128 --lr 0.1 --weight-decay 5e-4 --standardize --nesterov --test-fil --pooling avg --seed 123 --split-layer 7 --bottleneck-dim 2 --jvp-parallelism 100 --save-model --jacloss-alpha 0.1
python3 train.py --dataset cifar10 --model resnet18 --activation gelu --lr 0.001 --epochs 20 --target-lb 10. --bs 128 --weight-decay 5e-4 --standardize --nesterov --test-fil --pooling avg --seed 123 --split-layer 7 --bottleneck-dim 2 --jvp-parallelism 100 --split-learning --encoder-file models/resnet18_cifar100_l7_b2_standardize-True_jacloss0.1_act_gelu_pool_avg_bs128_seed123.pt --validate
```

### Score matching model training and evaluation
```bat
cd sliced_score_matching
python3 main.py --runner NICERunner --config nice/nice_ssm_vr.yml
python3 main.py --runner NICERunner --config nice/nice_cifar_ssm_vr.yml
python3 evaluate_scores.py
```

### Citation
If you use our work, please cite us:
```
@inproceedings{maeng2023bounding,
  author={Maeng, Kiwan and Guo, Chuan and Kariyappa, Sanjay and Suh, G. Edward},
  title={Bounding the Invertibility of Privacy-Preserving Instance Encoding Using Fisher Information},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023},
}
```

## License
The code is MIT licensed, as found in the LICENSE file.
