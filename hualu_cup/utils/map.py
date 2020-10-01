import torch
from torch.nn.functional import softmax


class MAP(object):
    def __init__(self):
        self.calling = SingleMAP()
        self.smoking = SingleMAP()
        self.normal = NormalMAP()

    def update_calling(self, predict, label):
        self.smoking.update(predict, label)
        self.normal.update(predict, label)

    def update_smoking(self, predict, label):
        self.calling.update(predict, label)
        self.normal.update(predict, label)

    def get(self):
        return (self.smoking.get() + self.normal.get() + self.calling.get()) / 3


class EvalMAP(object):
    def __init__(self):
        self.abnormal = SingleMAP()
        self.normal = NormalMAP()

    def update(self, predict, label):
        self.abnormal.update(predict, label)
        self.normal.update(predict, label)

    def get(self):
        return (self.abnormal.get() + self.normal.get()) / 2


class SingleMAP(object):
    def __init__(self):
        self.record = []
        self.total = 0
        self.ground_truth = 0

    def update(self, predict, label):
        predict = softmax(predict, -1)
        predict_label = torch.argmax(predict)
        tp = predict_label == label
        self.total += 1
        self.ground_truth += label
        self.record.append([predict[1], tp])

    def voc2010(self, pr):
        last_max = 0
        last_recall = 0
        ap = 0
        for i in range(pr.size(0)):
            if pr[i, 0] < last_max:
                pr[i, 0] = last_max
            elif i == pr.size(0) - 1:
                continue
            else:
                last_max = torch.max(pr[i:, 0])
            pr[i, 0] = last_max
            ap += (pr[i, 0] - last_recall) * pr[i, 0]
            last_recall = pr[i, 1]
        return ap

    def get(self):
        record = sorted(self.record, key=lambda x: x[0], reverse=True)
        sum_tp = 0
        pr = torch.zeros((len(record), 2))
        for i in range(len(record)):
            sum_tp += record[i][1]
            pr[i, 0] = float(sum_tp) / float(self.total) # precision
            pr[i, 1] = float(sum_tp) / float(self.ground_truth)  # recall
        ap = self.voc2010(pr)
        return ap


class NormalMAP(object):
    def __init__(self):
        self.record = []
        self.total = 0
        self.ground_truth = 0

    def update(self, predict, label):
        predict = softmax(predict, -1)
        predict_label = torch.argmax(predict)
        tp = predict_label == label
        self.total += 1
        self.ground_truth += (1 - label)
        self.record.append([predict[0], tp])

    def voc2010(self, pr):
        last_max = 0
        last_recall = 0
        ap = 0
        for i in range(pr.size(0)):
            if pr[i, 0] < last_max:
                pr[i, 0] = last_max
            elif i == pr.size(0) - 1:
                continue
            else:
                last_max = torch.max(pr[i:, 0])
            pr[i, 0] = last_max
            ap += (pr[i, 0] - last_recall) * pr[i, 0]
            last_recall = pr[i, 1]
        return ap

    def get(self):
        record = sorted(self.record, key=lambda x: x[0], reverse=True)
        sum_tp = 0
        pr = torch.zeros((len(record), 2))
        for i in range(len(record)):
            sum_tp += record[i][1]
            pr[i, 0] = float(sum_tp) / float(self.total)  # precision
            pr[i, 1] = float(sum_tp) / float(self.ground_truth)  # recall
        ap = self.voc2010(pr)
        return ap