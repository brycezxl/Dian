CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnet18 --lr 3e-5 --batch-size 8 | tee -a log/tmp/log.txt
