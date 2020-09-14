import numpy as np
import torch
import dataset
import argparse
import resnet
import config
import os
from time import time
from tensorboardX import SummaryWriter
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment
from tqdm import tqdm
#torch.cuda.set_device(0)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='train/evaluate/test')
    parser.add_argument('--task', type=str, help='call/safety')
    return parser.parse_args()

def evaluate(test_loader, epoch, model):
    model.eval()
    test_loss, correct, total, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()

    # print(len(test_loader))
    for data in tqdm(test_loader):
        try:
            image = data["image"].cuda()
            label = data["label"].cuda()
        except(OSError):
            # print("OSError of image. ")
            continue

        y_pred_raw, _, attention_map = model(image)
        crop_images = batch_augment(image, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
        y_pred_crop, _, _ = model(crop_images)
        y_pred = (y_pred_raw + y_pred_crop) / 2.

        loss = criterion(y_pred, label)
        test_loss += loss.item()

        _, predict = y_pred.max(1)
        total += label.size(0)
        correct += predict.eq(label).sum().item()
        tp += torch.sum(predict&label)
        fp += torch.sum(predict&(1-label))
        tn += torch.sum((1-predict)&(1-label))
        fn += torch.sum((1-predict)&label)
    acc = 100.*correct/total
    precision = 100.0*tp/(tp+fp).float()
    recall = 100.0*tp/(tp+fn).float()
    print("==> [evaluate] epoch {}, loss = {}, acc = {}, precision = {}, recall = {}".format(epoch, test_loss, acc, precision, recall))
    return acc, precision, recall

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

def model_train(model, input, eval_loader, test_loader, cfg):
    model.train()
    writer = SummaryWriter(log_dir=cfg.log_path)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005, amsgrad=True)
    current_epoch = 0
    global_step = 0
    if cfg.load_ckp == True:
        model, optimizer, current_epoch, global_step, loss = load_model(model, optimizer, cfg.ckp_path)

    feature_center = torch.zeros(2, cfg.num_attentions * model.num_features)
    center_loss = CenterLoss()

    for epoch in range(current_epoch, cfg.NUM_EPOCHS, 1):
        running_loss = 0.0
        _time = time()
        for i, data in enumerate(tqdm(input)):
            if i == len(input)-1: 
                break
            try:
                image = data["image"].cuda()
                label = data["label"].cuda()
            except(OSError):
                print("OSError of image. ")
                continue
            optimizer.zero_grad()
            y_pred_raw, feature_matrix, attention_map = model(image)
            
            '''
            # Update Feature Center            
            feature_center_batch = torch.nn.functional.normalize(feature_center[label], dim=-1)
            print(feature_center[label].shape, feature_matrix.detach().shape, feature_center_batch.shape)
            feature_center_batch[label] += cfg.beta * (feature_matrix.detach() - feature_center_batch)
            '''

            # Attention Cropping
            with torch.no_grad():
                crop_images = batch_augment(image, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)

            # crop images forward
            y_pred_crop, _, _ = model(crop_images)

            '''
            # Attention Dropping
            with torch.no_grad():
                drop_images = batch_augment(image, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))

            # drop images forward
            y_pred_drop, _, _ = model(drop_images)
            '''

            loss = criterion(y_pred_raw, label) / 3. + \
                         criterion(y_pred_crop, label) / 3
                         #criterion(y_pred_drop, label) / 3. + \
                         #0  #center_loss(feature_matrix, feature_center_batch)

            #print(loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0:
                batch_time = time() - _time
                print("==> [train] epoch {}, batch {}, global_step {}. loss for 10 batches: {}, time for 10 batches: {}s".format(epoch, i, global_step, running_loss, batch_time))
                writer.add_scalar("scalar/loss", running_loss, global_step, time())            
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
            }, os.path.join(cfg.save_path, "train_epoch_" + str(epoch) + ".tar"))
            print("==> [eval] on train dataset")
            acc_on_train, precision_on_train, recall_on_train = evaluate(train_loader, epoch, model)
            print("==> [eval] on valid dataset")
            acc_on_valid, precision_on_valid, recall_on_valid = evaluate(eval_loader, epoch, model)
            writer.add_scalar("scalar/accuracy_on_train", acc_on_train, global_step, time())
            writer.add_scalar("scalar/accuracy_on_valid", acc_on_valid, global_step, time())
            writer.add_scalar("scalar/precisoin_on_train", precision_on_train, global_step, time())
            writer.add_scalar("scalar/precision_on_valid", precision_on_valid, global_step, time())
            writer.add_scalar("scalar/recall_on_train", recall_on_train, global_step, time())
            writer.add_scalar("scalar/recall_on_valid", recall_on_valid, global_step, time())

    writer.close()
    print("Finish training.")

# TODO automatically log train process to local file / tensorboard
if __name__ == "__main__":
    
    args = init_args()

    if args.task == "call": 
        cfg = config.Config_call()
    elif args.task == "safety": 
        cfg = config.Config_safety()
    #print(cfg.ckp_path)
    from wsdan import WSDAN
    model = WSDAN(num_classes=2, M=32, net='inception_mixed_6e', pretrained=False).cuda()
    #model = resnet.model_init(device_ids=cfg.device_ids)
    if args.mode == "train":
        model.train()
        train_dataset = dataset.build_dataset_train(cfg.data_path, cfg.input_size)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, drop_last=True)
        eval_dataset = dataset.build_dataset_eval(cfg.data_path, cfg.input_size)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, drop_last=True)
        test_loader = None
        model_train(model, train_loader, eval_loader, test_loader, cfg)

    if args.mode == "eval":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg.lr)
        model, optimizer, epoch, global_step, loss = load_model(model, optimizer, cfg.ckp_path)
        eval_dataset = dataset.build_dataset_eval(cfg.data_path, cfg.input_size)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.batch_size, shuffle=True)
        #model.eval()
        evaluate(eval_loader, 0, model)

    if args.mode == "test":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg.lr)
        model, optimizer, epoch, global_step, loss = load_model(model, optimizer, cfg.ckp_path)
        test_dataset = dataset.build_dataset_test(cfg.data_path, cfg.input_size)
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
            print(output)

    if args.mode == "load":
        ckp_path = "/home/rlee/pic_project/ckp/train_epoch_25.tar"
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        model, optimizer, epoch, global_step, loss = load_model(model, optimizer, ckp_path)
        model.cpu()
        x = torch.randn((1, 1, 32, 32))
        torch_out = torch.onnx._export(model, x, "model.onnx", export_params=True)
