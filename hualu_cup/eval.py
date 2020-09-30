import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm

from utils.utils import batch_augment
from utils.utils import load_model
from torch import optim


def evaluate(data_loader, model, calling_ckp_path, smoking_ckp_path):

    with torch.no_grad():
        
        optimizer = torch.optim.Adagrad(model.parameters())
        model_calling, _, _, _, _ = load_model(model, optimizer, calling_ckp_path)
        model_smoking, _, _, _, _ = load_model(model, optimizer, smoking_ckp_path)
        model_calling.eval()
        model_smoking.eval()

        test_loss, correct, total, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0, 0
        criterion = torch.nn.CrossEntropyLoss()

        for data in tqdm(data_loader):
            image = data["image"].cuda()
            label = data["label"].cuda()

            y_pred_raw, _, attention_map = model(image)
            crop_images = batch_augment(image, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop, _, _ = model(crop_images)
            y_pred = (y_pred_raw + y_pred_crop) / 2.

            loss_ = criterion(y_pred, label)
            test_loss += loss_.item()

            _, predict = y_pred.max(1)
            total += label.size(0)
            correct += predict.eq(label).sum().item()
            tp += torch.sum(predict & label)
            fp += torch.sum(predict & (1 - label))
            tn += torch.sum((1 - predict) & (1 - label))
            fn += torch.sum((1 - predict) & label)
        acc = 100. * correct / total
        precision = 100.0 * tp / float(tp + fp)
        recall = 100.0 * tp / float(tp + fn)
        print("==> [evaluate] loss = {}, acc = {}, precision = {}, recall = {}".format(test_loss, acc,
                                                                                                 precision, recall))
    return acc, precision, recall
