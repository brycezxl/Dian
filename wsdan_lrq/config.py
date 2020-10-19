import torch                                                                                          
import time
import os     
import re
from glob import glob                                                                                       
                                                                                                      
class Config_call(object):                                                                                
    def __init__(self):      
        self.device_ids = [0]
        self.input_size = (512, 385)
        self.batch_size = 4
        self.USE_CUDA         =   torch.cuda.is_available()                                     
        self.NUM_EPOCHS       =   1000   
        self.num_workers      =   16
        self.evaluate_epoch   =   1 
        self.lr               =   1e-4
        self.root_dir         =   "/home/lrq/call/"
        self.save_path        =   "./ckp_call/0716/"   
        self.num_attentions = 8         # number of attention maps     
        self.beta = 5e-2                 # param for update feature centers  
        if not os.path.exists(self.save_path): 
            os.system("mkdir " + self.save_path)
        self.log_path         =   "./log_call/0716/"
        if not os.path.exists(self.log_path): 
            os.system("mkdir " + self.log_path)
        self.ckp_path         =   "./ckp_call/0629/train_epoch_3.tar"    
        self.data_path        =   "/data/Traffic_datas/1223_call/"
        #self.ckp_path = sorted(glob(os.path.join(self.save_path, '*')), key = lambda s:int(re.findall('(\d+)', s)[0]), reverse=True)[0]
        #self.ckp_path = None
        self.load_ckp = False
    # set learning rate strategy
    def get_lr(self, epoch):
        return 0.1
    
    # save all config to txt file
    def save_config_to_local_file(self):
        pass

class Config_safety(object):                                                                                
    def __init__(self):      
        self.device_ids = [1]
        self.input_size = (512, 385)
        self.batch_size = 4
        self.USE_CUDA         =   torch.cuda.is_available()                                     
        self.NUM_EPOCHS       =   1000   
        self.num_workers      =   16
        self.evaluate_epoch   =   1 
        self.lr               =   1e-4
        self.root_dir         =   "/home/lrq/call/"
        self.save_path        =   "/home/lrq/call/ckp_safety/0427/"       
        if not os.path.exists(self.save_path): 
            os.system("mkdir " + self.save_path)
        self.log_path         =   "./log_safety/"
        if not os.path.exists(self.log_path): 
            os.system("mkdir " + self.log_path)
        self.ckp_path         =   "/home/lrq/call/ckp_safety/0427/train_epoch_22.tar"   
        #self.data_path        =   "/data/Traffic_datas/6011TrainAndValid/"
        self.data_path        =   "/reserve/liruiqi/docunet/"
        '''
        try: 
            self.ckp_path = sorted(glob(os.path.join(self.save_path, '*')), key = lambda s:int(re.findall('(\d+)', s)[0]), reverse=True)[0]
        except: 
            self.ckp_path = None
        '''
        self.load_ckp = True
    # set learning rate strategy
    def get_lr(epoch):
        return 0.1
    
    # save all config to txt file
    def save_config_to_local_file():
        pass

