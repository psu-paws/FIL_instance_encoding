CUDA_VISIBLE_DEVICES=1 python3 train.py --dataset movielens-20 --model ncf-mlp --bs 128 --lr 0.1 --eval-every 1 --epochs 1 --metrics "loss,auc" --test-fil --jvp-parallelism 100 --split-layer 0 --scheduler "none" --train-lb 1. --seed 123 --print-every 1410 --optimizer adagrad --save-model --jacloss-alpha 0.01
CUDA_VISIBLE_DEVICES=1 python3 train.py --dataset movielens-20 --model ncf-mlp --bs 128 --lr 0.1 --eval-every 1 --epochs 1 --metrics "loss,auc" --test-fil --jvp-parallelism 100 --split-layer 0 --scheduler "none" --train-lb 10. --seed 123 --print-every 1410 --optimizer adagrad --save-model
CUDA_VISIBLE_DEVICES=1 python3 train.py --dataset movielens-20 --model ncf-mlp --bs 128 --lr 0.1 --eval-every 1 --epochs 1 --metrics "loss,auc" --test-fil --jvp-parallelism 100 --split-layer 0 --scheduler "none" --train-lb 100. --seed 123 --print-every 1410 --optimizer adagrad --save-model