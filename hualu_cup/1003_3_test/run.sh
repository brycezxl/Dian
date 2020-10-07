CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnet18 --lr 1e-4 --batch-size 16 | tee -a log/tmp/log_res18_1.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnet18 --lr 3e-5 --batch-size 16 | tee -a log/tmp/log_res18_2.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnet18 --lr 1e-5 --batch-size 16 | tee -a log/tmp/log_res18_3.txt

CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext_wsl --lr 1e-4 --batch-size 16 | tee -a log/tmp/log_wsl_1.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext_wsl --lr 3e-5 --batch-size 16 | tee -a log/tmp/log_wsl_2.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext_wsl --lr 1e-5 --batch-size 16 | tee -a log/tmp/log_wsl_3.txt

CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext101 --lr 1e-4 --batch-size 16 | tee -a log/tmp/log_resnext101_1.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext101 --lr 3e-5 --batch-size 16 | tee -a log/tmp/log_resnext101_2.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model resnext101 --lr 1e-5 --batch-size 16 | tee -a log/tmp/log_resnext101_3.txt