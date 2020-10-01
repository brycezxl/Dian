import os

import torch
from torch import cuda


class Config(object):
    def __init__(self, args):

        self.batch_size = 4
        self.lr = 1e-4
        self.num_attentions = 8          # number of attention maps
        self.beta = 5e-2                 # param for update feature centers

        self.input_size = (608, 480)
        self.USE_CUDA = torch.cuda.is_available()
        self.print_interval = 200
        self.device_ids = [0]
        self.num_workers = 16
        self.NUM_EPOCHS = 15

        self.save_path = "./log/" + args.task + "/tmp"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.data_path = "./data/"
        self.calling_ckp_path = "./log/calling/train.tar"
        self.smoking_ckp_path = "./log/smoking/train.tar"
