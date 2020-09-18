import os

import torch
from torch import cuda


class ConfigCalling(object):
    def __init__(self):

        self.batch_size = 4
        self.lr = 1e-4
        self.num_attentions = 8          # number of attention maps
        self.beta = 5e-2                 # param for update feature centers

        self.input_size = (512, 385)
        self.USE_CUDA = torch.cuda.is_available()
        self.print_interval = 1000
        self.device_ids = [0]
        self.num_workers = 16
        self.NUM_EPOCHS = 10

        self.save_path = "./log/"
        if not os.path.exists(self.save_path): 
            os.system("mkdir " + self.save_path)
        self.data_path = "./data/"
        self.ckp_path = "./log/ckp.tar"
