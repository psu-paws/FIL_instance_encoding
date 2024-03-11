#CUDA_VISIBLE_DEVICES=1 python3 run_glue.py --split-layer 0 --train-lb 10 --jacloss-alpha 0.01
#CUDA_VISIBLE_DEVICES=1 python3 run_glue.py --split-layer 0 --train-lb 100 --jacloss-alpha 0.01
#CUDA_VISIBLE_DEVICES=1 python3 run_glue.py --split-layer 2 --train-lb 10 --jacloss-alpha 1.
CUDA_VISIBLE_DEVICES=1 python3 run_glue.py --split-layer 2 --train-lb 100 --jacloss-alpha 0.
#CUDA_VISIBLE_DEVICES=1 python3 run_glue.py --split-layer 4 --train-lb 10 --jacloss-alpha 1.
CUDA_VISIBLE_DEVICES=1 python3 run_glue.py --split-layer 4 --train-lb 100 --jacloss-alpha 0.
