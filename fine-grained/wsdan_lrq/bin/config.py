import torch                                                                                          
import time
import os     
import re
from glob import glob                                                                                       
                                                                                                      
class Config(object):                                                                                
    def __init__(self):      
        self.input_size = (256, 512)
        self.batch_size = 16
        self.USE_CUDA         =   torch.cuda.is_available()                                     
        self.NUM_EPOCHS       =   1000   
        self.evaluate_epoch   =   1 
        self.lr               =   1e-3
        self.root_dir         =   "/mnt/data/lrq/resnet_binary_classification/"
        self.save_path        =   "./model_save_files/"            
        self.log_path = "./log"
        self.ckp_path = sorted(glob(os.path.join(self.save_path, '*')), key = lambda s:int(re.findall('(\d+)', s)[0]), reverse=True)[0]
        self.load_ckp = True
    # set learning rate strategy
    def get_lr(epoch):
        return 0.1
    
    # save all config to txt file
    def save_config_to_local_file():
        pass
