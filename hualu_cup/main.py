import argparse

import torch.utils.data
from torch import optim

import config
from eval import evaluate
from models import *
from train import train
from utils import data_loader
from utils.utils import load_model


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train", help='train/evaluate/test')
    parser.add_argument('--task', type=str, default="call", help='call/safety')
    return parser.parse_args()


# TODO automatically log train process to local file / tensorboard
if __name__ == "__main__":

    args = init_args()

    if args.task == "call":
        cfg = config.ConfigCall()
    elif args.task == "safety":
        cfg = config.ConfigSafety()
    else:
        raise ValueError("Task must be call or safety!")

    print("Loading data")
    model = WSDAN(num_classes=2, M=32, net='inception_mixed_6e', pretrained=False).cuda()
    # model = resnet.model_init(device_ids=cfg.device_ids)

    if args.mode == "train":
        model.train()
        train_dataset = data_loader.build_dataset_train(cfg.data_path, cfg.input_size)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size,
                                                   num_workers=cfg.num_workers, shuffle=True, drop_last=True)
        eval_dataset = data_loader.build_dataset_eval(cfg.data_path, cfg.input_size)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.batch_size,
                                                  num_workers=cfg.num_workers, shuffle=True, drop_last=True)
        train(model, train_loader, eval_loader, cfg)

    if args.mode == "eval":
        model.eval()
        optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg.lr)
        model, optimizer, epoch, global_step, loss = load_model(model, optimizer, cfg.ckp_path)
        eval_dataset = data_loader.build_dataset_eval(cfg.data_path, cfg.input_size)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.batch_size, shuffle=True)
        evaluate(eval_loader, 0, model)

    if args.mode == "test":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg.lr)
        model, optimizer, epoch, global_step, loss = load_model(model, optimizer, cfg.ckp_path)
        test_dataset = data_loader.build_dataset_test(cfg.data_path, cfg.input_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)
        model.eval()
        for i, data in enumerate(test_loader):
            try:
                image = data["image"].cuda()
                label = data["label"].cuda()
            except OSError:
                print("OSError of image. ")
                continue
            output = model(image)
            print(output)
