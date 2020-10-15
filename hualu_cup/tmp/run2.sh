CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model wsdan --lr 1e-4 --batch-size 8 | tee -a log/tmp/ws+nest1.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model wsdan --lr 3e-4 --batch-size 8 | tee -a log/tmp/ws+nest2.txt
