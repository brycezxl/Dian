CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext101 --lr 8e-5 --batch-size 8 | tee -a log/tmp/log_101_1.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext101 --lr 4e-5 --batch-size 8 | tee -a log/tmp/log_101_2.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext101 --lr 2e-5 --batch-size 8 | tee -a log/tmp/log_101_3.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext101 --lr 1e-5 --batch-size 8 | tee -a log/tmp/log_101_4.txt

CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext_wsl --lr 8e-5 --batch-size 8 | tee -a log/tmp/log_wsl_1.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext_wsl --lr 4e-5 --batch-size 8 | tee -a log/tmp/log_wsl_2.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext_wsl --lr 2e-5 --batch-size 8 | tee -a log/tmp/log_wsl_3.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext_wsl --lr 1e-5 --batch-size 8 | tee -a log/tmp/log_wsl_4.txt
