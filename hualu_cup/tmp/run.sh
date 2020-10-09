CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --lr 2e-5 --batch-size 8 | tee -a log/tmp/101_1.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --lr 1e-5 --batch-size 8 | tee -a log/tmp/101_2.txt

CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_wsl --lr 2e-5 --batch-size 8 | tee -a log/tmp/wsl_1.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_wsl --lr 1e-5 --batch-size 8 | tee -a log/tmp/wsl_2.txt
