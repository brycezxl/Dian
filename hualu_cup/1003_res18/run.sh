CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext_wsl --lr 3e-5 --batch-size 8 | tee -a log/tmp/log_wsl1.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext_wsl --lr 1e-4 --batch-size 8 | tee -a log/tmp/log_wsl2.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext_wsl --lr 3e-5 --batch-size 16 | tee -a log/tmp/log_wsl3.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext_wsl --lr 1e-4 --batch-size 16 | tee -a log/tmp/log_wsl4.txt
