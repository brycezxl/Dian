import os
from time import time

import torch
import torch.utils.data
from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm import tqdm

from eval import evaluate
from utils.utils import batch_augment


def train(model, train_loader, eval_loader, cfg):
    model.train()
    print("Start training")
    writer = SummaryWriter(log_dir=cfg.save_path)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005, amsgrad=True)
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
            y_pred_raw, feature_matrix, attention_map = model(image)

            '''
            # Update Feature Center            
            feature_center_batch = torch.nn.functional.normalize(feature_center[label], dim=-1)
            print(feature_center[label].shape, feature_matrix.detach().shape, feature_center_batch.shape)
            feature_center_batch[label] += cfg.beta * (feature_matrix.detach() - feature_center_batch)
            '''

            # Attention Cropping
            with torch.no_grad():
                crop_images = batch_augment(image, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6),
                                            padding_ratio=0.1)

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
            #   criterion(y_pred_drop, label) / 3. + \
            #   0 center_loss(feature_matrix, feature_center_batch)

            # print(loss)
            loss.backward()
            optimizer.step()

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
