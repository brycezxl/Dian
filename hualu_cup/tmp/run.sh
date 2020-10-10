CUDA_VISIBLE_DEVICES=0 python3 main.py --mode train --model res_18 --lr 4e-5 --batch-size 8 | tee -a log/tmp/18_1.txt
CUDA_VISIBLE_DEVICES=0 python3 main.py --mode train --model res_18 --lr 2e-5 --batch-size 8 | tee -a log/tmp/18_2.txt
CUDA_VISIBLE_DEVICES=0 python3 main.py --mode train --model res_18 --lr 8e-5 --batch-size 8 | tee -a log/tmp/18_3.txt
