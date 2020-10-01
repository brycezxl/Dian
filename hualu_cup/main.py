import argparse

import torch.utils.data

from config import Config
from eval import evaluate
from models import *
from test import test
from train import train
from utils import data_loader
from torch import nn


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train", help='train/test')
    parser.add_argument('--task', type=str, default="calling", help='calling/smoking')
    return parser.parse_args()


# TODO automatically log train process to local file / tensorboard
if __name__ == "__main__":

    args = init_args()
    cfg = Config(args)

    print("Start loading data")
    # model = WSDAN(num_classes=2, M=32, net='inception_mixed_6e', pretrained=False).cuda()
    model = resnext101_32x8d(pretrained=True, progress=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, 2)
    )
    model.cuda()

    if args.mode == "train":
        train_dataset = data_loader.build_dataset_train(cfg.data_path, cfg.input_size, args.task)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size,
                                                   num_workers=cfg.num_workers, shuffle=True, drop_last=True)
        eval_dataset = data_loader.build_dataset_eval(cfg.data_path, cfg.input_size, args.task)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.batch_size,
                                                  num_workers=cfg.num_workers, shuffle=True, drop_last=True)
        train(model, train_loader, eval_loader, cfg)

    if args.mode == "eval":
        eval_dataset = data_loader.build_dataset_eval(cfg.data_path, cfg.input_size, None)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.batch_size,
                                                  num_workers=cfg.num_workers, shuffle=True, drop_last=True)
        evaluate(eval_loader, model, cfg.calling_ckp_path, cfg.smoking_ckp_path)

    if args.mode == "test":
        test_dataset = data_loader.build_dataset_test(cfg.data_path, cfg.input_size, None)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

        test(test_loader, model, cfg.calling_ckp_path, cfg.smoking_ckp_path)
