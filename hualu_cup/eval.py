import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm
from utils.map import MAP, EvalMAP
from utils.utils import batch_augment
from utils.utils import load_model
from torch import optim
from torch.nn.functional import softmax


def evaluate(data_loader, epoch, model):
    with torch.no_grad():

        # meter = EvalMAP()

        test_loss, correct, total, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0, 0
        criterion = torch.nn.CrossEntropyLoss()

        for data in tqdm(data_loader):
            image = data["image"].cuda()
            label = data["label"].cuda()

            # call
            y_pred = model(image)
            loss = criterion(y_pred, label) / 2
            # predict = softmax(y_pred, dim=-1)
            # for i in range(predict.size(0)):
            #     meter.update(predict[i, :], label[i])

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
        # m_ap = meter.get()
        print("==> [evaluate] loss = %.3f, acc = %.3f, precision = %.3f, recall = %.3f"
              "" % (loss, acc, precision, recall))
    return acc, precision, recall