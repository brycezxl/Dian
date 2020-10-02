import math
import os
from time import time

import torch
import torch.utils.data
from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm

from eval import evaluate
from models.label_smooth_ce_loss import LabelSmoothCELoss


def train(model, train_loader, eval_loader, args):
    model.train()
    print("Start training")
    writer = SummaryWriter(log_dir=args.save_path)

    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005, amsgrad=True)

    criterion = LabelSmoothCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005, amsgrad=True)

    epochs = 15
    warm_up_epochs = 5
    warm_up_with_cosine_lr = lambda e: e / warm_up_epochs if e <= warm_up_epochs else 0.5 * (
            math.cos((e - warm_up_epochs) / (epochs - warm_up_epochs) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)  # CosineAnnealingLR

    current_epoch = 0
    global_step = 0
    best_map = 0
    loss = 0

    for epoch in range(current_epoch, args.num_epochs, 1):
        running_loss = 0.0
        t = time()
        for i, data in enumerate(tqdm(train_loader)):
            image = data["image"].cuda()
            label = data["label"].cuda()

            optimizer.zero_grad()

            # y_pred_raw, feature_matrix, attention_map = model(image)
            # with torch.no_grad(): # Attention Cropping
            #     crop_images = batch_augment(image, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6),
            #                                 padding_ratio=0.1)
            # y_pred_crop, _, _ = model(crop_images)  # crop images forward
            # loss = criterion(y_pred_raw, label) / 3. + \
            #     criterion(y_pred_crop, label) / 3

            predict = model(image)
            loss = criterion(predict, label)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % args.print_interval == 0 and i != 0:
                batch_time = (time() - t) / args.print_interval / args.batch_size
                running_loss = running_loss / args.print_interval
                print("==> [train] epoch = %2d, batch = %4d, global_step = %4d, loss = %.2f, "
                      "time per picture = %.2fs" % (epoch, i, global_step, running_loss, batch_time))
                writer.add_scalar("scalar/loss", running_loss, global_step, time())
                running_loss = 0.0
                t = time()
            global_step += 1

        print("==> [train] epoch = %2d, loss = %.2f, time per picture = %.2fs"
              % (epoch + 1, running_loss / len(train_loader),
                 (time() - t) / len(train_loader) / args.batch_size))
        print("==> [eval on train dataset] ", end='')
        map_on_train, acc_on_train, precision_on_train, recall_on_train = evaluate(train_loader, model)
        print("==> [eval on valid dataset] ", end='')
        map_on_valid, acc_on_valid, precision_on_valid, recall_on_valid = evaluate(eval_loader, model)

        writer.add_scalar("scalar/mAP_on_train", map_on_train, global_step, time())
        writer.add_scalar("scalar/mAP_on_train", map_on_valid, global_step, time())
        writer.add_scalar("scalar/accuracy_on_train", acc_on_train, global_step, time())
        writer.add_scalar("scalar/accuracy_on_valid", acc_on_valid, global_step, time())
        writer.add_scalar("scalar/precision_on_train", precision_on_train, global_step, time())
        writer.add_scalar("scalar/precision_on_valid", precision_on_valid, global_step, time())
        writer.add_scalar("scalar/recall_on_train", recall_on_train, global_step, time())
        writer.add_scalar("scalar/recall_on_valid", recall_on_valid, global_step, time())

        if map_on_valid > best_map:
            best_map = map_on_valid
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                'loss': loss,
            }, os.path.join(args.save_path, "%.4f" % best_map + ".tar"))
            print("==> [best] mAP: %.3f" % best_map)

    writer.close()
    print("Finish training.")
