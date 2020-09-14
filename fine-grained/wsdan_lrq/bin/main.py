import numpy as np
import torch
import torchvision
import dataset
import argparse
import model
from config import Config
from time import time
from tensorboardX import SummaryWriter

torch.cuda.set_device(0)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='train/evaluate/test')
    return parser.parse_args()

def model_init(num_classes):
    model = torchvision.models.resnet50(num_classes=2)
    model.cuda()
    return model

def evaluate(epoch, model):
    cfg = Config()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    test_dataset = dataset.build_dataset(cfg.root_dir, cfg.input_size, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)
    # print(len(test_loader))
    for data in test_loader:
        try:
            image = data["image"].cuda()
            label = data["label"].cuda()
        except(OSError):
            # print("OSError of image. ")
            continue

        outputs = model(image)
        loss = criterion(outputs, label)
        test_loss += loss.item()

        _, predicted = outputs.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        # progress_bar(i, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(i+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    print("==> [evaluate] epoch {}, loss = {}, acc = {}".format(epoch, test_loss, acc))
    return acc

def load_model(model, optimizer, ckp_path):
    checkpoint = torch.load(ckp_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    try:
        global_step = checkpoint["global_step"]
    except:
        global_step = 0
    return model, optimizer, epoch, global_step, loss

def model_train(model, input, num_epoch, save_path, ckp_path, log_path, load_ckp=False):
    cfg = Config()
    model.train()
    writer = SummaryWriter(log_dir=log_path)

    # TODO try new loss function ad optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)

    current_epoch = 0
    global_step = 0
    if load_ckp == True:
        model, optimizer, current_epoch, global_step, loss = load_model(model, optimizer, ckp_path)

    # TODO add tensorboard to visualize training process
    for epoch in range(current_epoch, num_epoch, 1):
        running_loss = 0.0
        _time = time()
        for i, data in enumerate(input):
            try:
                image = data["image"].cuda()
                label = data["label"].cuda()
            except(OSError):
                print("OSError of image. ")
                continue
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() 
            if i % 10 == 0:
                batch_time = time() - _time
                print("==> [train] epoch {}, batch {}, global_step {}. loss for 10 batches: {}, time for 10 batches: {}s".format(epoch, i, global_step, running_loss, batch_time))
                writer.add_scalar("scalar/loss", running_loss)            
                running_loss = 0.0
                _time = time()
            global_step += 1

        # TODO add save condition eg. acc

        if epoch % cfg.evaluate_epoch == 0:
            torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            'loss': loss,
            }, save_path + "train_epoch_" + str(epoch) + ".tar")
            acc = evaluate(epoch, model)
            writer.add_scalar("scalar/accuracy", acc, global_step, time())

    writer.close()
    print("Finish training.")

# TODO automatically log train process to local file / tensorboard
if __name__ == "__main__":
    cfg = Config()
    args = init_args()

    model = model.model_init(num_classes=2)
    if args.mode == "train":
        model.train()
        train_dataset = dataset.build_dataset(cfg.root_dir, cfg.input_size, args.mode)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        model_train(model, train_loader, cfg.NUM_EPOCHS, cfg.save_path, cfg.ckp_path, cfg.log_path, cfg.load_ckp)

    if args.mode == "test":
        test_dataset = dataset.build_dataset(cfg.root_dir, cfg.input_size, args.mode)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)
        model.eval()
        for i, data in enumerate(test_loader):
            try:
                image = data["image"].cuda()
                label = data["label"].cuda()
            except(OSError):
                print("OSError of image. ")
                continue

            output = model(image)
