import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm

from utils.utils import batch_augment


def evaluate(test_loader_, epoch_, model_):
    model_.eval()
    test_loss, correct, total, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()

    # print(len(test_loader))
    for data_ in tqdm(test_loader_):
        try:
            image_ = data_["image"].cuda()
            label_ = data_["label"].cuda()
        except OSError:
            # print("OSError of image. ")
            continue

        y_pred_raw, _, attention_map = model_(image_)
        crop_images = batch_augment(image_, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
        y_pred_crop, _, _ = model_(crop_images)
        y_pred = (y_pred_raw + y_pred_crop) / 2.

        loss_ = criterion(y_pred, label_)
        test_loss += loss_.item()

        _, predict = y_pred.max(1)
        total += label_.size(0)
        correct += predict.eq(label_).sum().item()
        tp += torch.sum(predict & label_)
        fp += torch.sum(predict & (1 - label_))
        tn += torch.sum((1 - predict) & (1 - label_))
        fn += torch.sum((1 - predict) & label_)
    acc = 100. * correct / total
    precision = 100.0 * tp / float(tp + fp)
    recall = 100.0 * tp / float(tp + fn)
    print("==> [evaluate] epoch {}, loss = {}, acc = {}, precision = {}, recall = {}".format(epoch_, test_loss, acc,
                                                                                             precision, recall))
    return acc, precision, recall
