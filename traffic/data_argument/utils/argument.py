import torch


def select_car(detections):
    left = detections[:, 0]
    down = detections[:, 1]
    right = detections[:, 2]
    up = detections[:, 3]
    area = (right - left) * (up - down)
    print(area)
    biggest = torch.argmax(area)
    return detections[biggest, :]
