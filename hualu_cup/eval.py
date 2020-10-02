import torch
import torch.utils.data
from torch import nn
from torch.nn.functional import softmax

from utils.voc2010 import VOCMApMetric


def evaluate(data_loader, model):

    with torch.no_grad():

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        meter = VOCMApMetric(class_names=("abnormal", "normal"))
        correct, total, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0

        for data in data_loader:
            image = data["image"].cuda()
            label = data["label"].cuda()

            predict = model(image)
            loss = criterion(predict, label)

            predict = softmax(predict, dim=-1)
            pred_labels = torch.argmax(predict, dim=-1)
            pred_bboxes = torch.ones((pred_labels.size(0), 1, 4))
            gt_bboxes = torch.ones((pred_labels.size(0), 1, 4))
            meter.update(pred_bboxes.cpu().numpy(), pred_labels.cpu().numpy(), predict.cpu().numpy(),
                         gt_bboxes.cpu().numpy(), label.cpu().numpy())

            _, predict = predict.max(1)
            total += label.size(0)
            correct += predict.eq(label).sum().item()
            tp += torch.sum(predict & label)
            fp += torch.sum(predict & (1 - label))
            tn += torch.sum((1 - predict) & (1 - label))
            fn += torch.sum((1 - predict) & label)

        acc = float(correct) / float(total)
        precision = (float(tp) + 1e-6) / (float(tp + fp) + 1e-3)
        recall = (float(tp) + 1e-6) / (float(tp + fn) + 1e-3)
        m_ap = meter.get()[1][2]
        print("mAP = %.3f, loss = %.3f, acc = %.3f, precision = %.3f, recall = %.3f"
              "" % (m_ap, loss, acc, precision, recall))
    return m_ap, acc, precision, recall
