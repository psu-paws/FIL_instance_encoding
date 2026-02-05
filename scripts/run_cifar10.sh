CUDA_VISIBLE_DEVICES=0 python3 run.py --jvp-parallelism 100 --split-layer 1 --target-lb 10.0 --dataset cifar10 --model resnet18 --train-seed 123 --test-seed 123 --activation relu --standardize --pooling max --bottleneck-dim 4 --train-lb 0.0 --jvp-parallelism 100 --jacloss-alpha 0.0 --load-from-file

