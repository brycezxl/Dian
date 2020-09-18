import os
from glob import glob
import random
import PIL
import torch
import torchvision
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BinaryClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, input_size, task, transforms=None, argument_path=None):
        self.task = task
        self.img_list = glob(os.path.join(img_path, "*", "*.jpg"))
        if argument_path:
            self.img_list += glob(os.path.join(argument_path, "*", "*.jpg"))
        if transforms is None:
            self.transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(input_size),
                                                              torchvision.transforms.ToTensor()])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        try:
            img = PIL.Image.open(self.img_list[idx])
            img = self.transforms(img)
        except OSError:
            print("OSError at combined_mask path ", self.img_list[idx])
            return None

        if self.task == 'calling':
            if self.img_list[idx].split('/')[-2] == "calling_images":
                label = 1
            else:
                label = 0
        else:
            raise EOFError

        sample = {"image": img, "label": label}
        return sample


def build_dataset_train(data_path, input_size, task):
    return BinaryClassifierDataset(os.path.join(data_path, "train"), input_size, task)


def build_dataset_eval(data_path, input_size, task):
    return BinaryClassifierDataset(os.path.join(data_path, "train"), input_size, task)


def build_dataset_test(data_path, input_size, task):
    return BinaryClassifierDataset(os.path.join(data_path, "test"), input_size, task)


def split_dataset():
    train_path = "./data/train/"
    eval_path = "./data/eval/"
    if os.path.exists(eval_path):
        os.remove(eval_path)
    os.mkdir(eval_path)
    for p in ["calling_images/", "normal_images/", "smoking_images/"]:
        img_list = glob(os.path.join(train_path + p, "*", "*.jpg"))
        random.shuffle(img_list)
        img_list = img_list[:int(0.8 * len(img_list))]
        for img in img_list:
            new_img = img
            shutil.move(img, new_img)


if __name__ == '__main__':
    split_dataset()