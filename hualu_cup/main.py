import argparse
import os

import torch.utils.data

from models import *
from test import test
from train import train
from utils import data_loader


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train", help='train/test')
    parser.add_argument('--task', type=str, default="calling", help='calling/smoking')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--model', type=str, default="resnet18")

    parser.add_argument('--print-interval', type=int, default=2000)
    parser.add_argument('--num-attentions', type=int, default=8)
    parser.add_argument('--input-size', type=tuple, default=(224, 224))
    parser.add_argument('--beta', type=float, default=5e-2)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--num-epochs', type=float, default=15)

    parser.add_argument('--data-path', type=str, default="./data/")
    parser.add_argument('--smoking-ckp-path', type=str, default="./log/smoking/train.tar")
    parser.add_argument('--calling-ckp-path', type=str, default="./log/calling/train.tar")
    parser.add_argument('--save-path', type=str, default="./log/tmp/")
    if not os.path.exists(parser.parse_args().save_path):
        os.mkdir(parser.parse_args().save_path)

    return parser.parse_args()


# TODO automatically log train process to local file / tensorboard
if __name__ == "__main__":

    args = init_args()

    print("Start loading data")

    if args.model == "wsdan":
        model = WSDAN(num_classes=2, M=32, net='inception_mixed_6e', pretrained=False)
    elif args.model == "resnext101":
        model = resnext101_32x8d(pretrained=True, progress=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 2)
        )
    elif args.model == "resnet18":
        model = resnet18(pretrained=True, progress=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    else:
        raise ValueError
    model.cuda()

    if args.mode == "train":
        train_dataset = data_loader.build_dataset_train(args.data_path, args.input_size, args.task)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True, drop_last=True)
        eval_dataset = data_loader.build_dataset_eval(args.data_path, args.input_size, args.task)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, shuffle=True, drop_last=True)
        train(model, train_loader, eval_loader, args)

    # if args.mode == "eval":
    #     eval_dataset = data_loader.build_dataset_eval(cfg.data_path, cfg.input_size, None)
    #     eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.batch_size,
    #                                               num_workers=cfg.num_workers, shuffle=True, drop_last=True)
    #     evaluate(eval_loader, model, cfg.calling_ckp_path, cfg.smoking_ckp_path)

    if args.mode == "test":
        test_dataset = data_loader.build_dataset_test(args.data_path, args.input_size, None)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        test(test_loader, model, args.calling_ckp_path, args.smoking_ckp_path)
