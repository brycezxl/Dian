import os
import random
import shutil
from glob import glob

import PIL
import cv2
import torch
import torchvision
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BinaryClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, input_size, task, mode=None, transforms=None, argument_path=None):
        self.mode = mode
        self.task = task
        if self.mode == 'test':
            self.img_list = glob(os.path.join(img_path, "*.jpg"))
        else:
            self.img_list = glob(os.path.join(img_path, "*", "*.jpg"))
        if transforms is None:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((input_size[0], input_size[1])),  # 等比填充缩放
                torchvision.transforms.RandomCrop(input_size[0], input_size[1]),
                torchvision.transforms.RandomHorizontalFlip(),
                # torchvision.transforms.RandomGaussianBlur(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        try:
            img = PIL.Image.open(self.img_list[idx])
            if img.layers == 1:
                img = cv2.imread(self.img_list[idx], 0)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = self.transforms(img)
        except OSError:
            print("OSError at combined_mask path ", self.img_list[idx])
            return None

        # test/1036.jpg
        if img.size(0) == 4:
            img = img[:3, :, :]

        if self.mode == 'train':
            if self.task == 'calling':
                if self.img_list[idx].split('/')[-2] == "calling_images":
                    label = 1
                else:
                    label = 0
            elif self.task == 'smoking':
                if self.img_list[idx].split('/')[-2] == "smoking_images":
                    label = 1
                else:
                    label = 0
            else:
                raise EOFError
            sample = {"image": img, "label": label}
        elif self.mode == 'eval':
            if self.img_list[idx].split('/')[-2] == "normal_images":
                label = 0
            else:
                label = 1
            sample = {"image": img, "label": label}
        else:
            sample = {"image": img, "name": self.img_list[idx].split('/')[-1]}

        return sample


def build_dataset_train(data_path, input_size, task):
    return BinaryClassifierDataset(os.path.join(data_path, "train"), input_size, task, mode='train')


def build_dataset_eval(data_path, input_size, task):
    return BinaryClassifierDataset(os.path.join(data_path, "eval"), input_size, task, mode='eval')


def build_dataset_test(data_path, input_size, task):
    return BinaryClassifierDataset(os.path.join(data_path, "test"), input_size, task, mode='test')


def split_dataset():
    train_path = "../data/train/"
    eval_path = "../data/eval/"
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
    if not os.path.exists(eval_path + "calling_images/"):
        os.mkdir(eval_path + "calling_images/")
    if not os.path.exists(eval_path + "normal_images/"):
        os.mkdir(eval_path + "normal_images/")
    if not os.path.exists(eval_path + "smoking_images/"):
        os.mkdir(eval_path + "smoking_images/")

    for p in ["calling_images/", "normal_images/", "smoking_images/"]:
        img_list = glob(os.path.join(train_path + p, "*.jpg"))
        random.shuffle(img_list)
        img_list = img_list[:int(0.2 * len(img_list))]
        for img in img_list:
            new_img = img.replace("train", "eval")
            shutil.move(img, new_img)


if __name__ == '__main__':
    split_dataset()
