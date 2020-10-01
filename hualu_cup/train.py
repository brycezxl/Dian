import os
from time import time
import math
import torch
import torch.utils.data
from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm import tqdm
from models.label_smooth_ce_loss import LabelSmoothCELoss
from eval import evaluate
from utils.utils import batch_augment


def train(model, train_loader, eval_loader, cfg):
    model.train()
    print("Start training")
    writer = SummaryWriter(log_dir=cfg.save_path)

    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005, amsgrad=True)

    criterion = LabelSmoothCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0005, amsgrad=True)

    epochs = 15
    warm_up_epochs = 5
    warm_up_with_cosine_lr = lambda e: e / warm_up_epochs if e <= warm_up_epochs else 0.5 * (
            math.cos((e - warm_up_epochs) / (epochs - warm_up_epochs) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)  # CosineAnnealingLR

    current_epoch = 0
    global_step = 0
    loss = 0
    # if cfg.load_ckp:
    #     model, optimizer, current_epoch, global_step, loss = load_model(model, optimizer, cfg.ckp_path)

    # feature_center = torch.zeros(2, cfg_.num_attentions * model_.num_features)
    # center_loss = CenterLoss()

    for epoch in range(current_epoch, cfg.NUM_EPOCHS, 1):
        running_loss = 0.0
        t = time()
        for i, data in enumerate(tqdm(train_loader)):
            if i == len(train_loader) - 1:
                break
            try:
                image = data["image"].cuda()
                label = data["label"].cuda()
            except OSError:
                print("OSError of image. ")
                continue
            optimizer.zero_grad()

            # y_pred_raw, feature_matrix, attention_map = model(image)
            # with torch.no_grad(): # Attention Cropping
            #     crop_images = batch_augment(image, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6),
            #                                 padding_ratio=0.1)
            # y_pred_crop, _, _ = model(crop_images)  # crop images forward
            # loss = criterion(y_pred_raw, label) / 3. + \
            #     criterion(y_pred_crop, label) / 3

            y = model(image)
            loss = criterion(y, label)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if i % cfg.print_interval == 0 and i != 0:
                batch_time = time() - t
                print("==> [train] epoch {}, batch {}, global_step {}. loss for 10 batches: {}, "
                      "time for 10 batches: {}s".format(epoch, i, global_step, running_loss, batch_time))
                writer.add_scalar("scalar/loss", running_loss, global_step, time())
                running_loss = 0.0
                t = time()
            global_step += 1

        # TODO add save condition eg. acc
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
        writer.add_scalar("scalar/precision_on_train", precision_on_train, global_step, time())
        writer.add_scalar("scalar/precision_on_valid", precision_on_valid, global_step, time())
        writer.add_scalar("scalar/recall_on_train", recall_on_train, global_step, time())
        writer.add_scalar("scalar/recall_on_valid", recall_on_valid, global_step, time())

    writer.close()
    print("Finish training.")
