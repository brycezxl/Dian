CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --lr 1e-5 --batch-size 8 | tee -a log/tmp/cbam_base_ad_1e-5.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --lr 2e-5 --batch-size 8 | tee -a log/tmp/cbam_base_ad_2e-5.txt
