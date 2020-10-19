import os
from glob import glob

import PIL
import torch
import torchvision
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BinaryClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, input_size=(32, 32), transforms=None, argument_path=None):
        self.img_list = glob(os.path.join(img_path, "*", "*.jpg"))
        if argument_path:
            self.img_list += glob(os.path.join(argument_path, "*", "*.jpg"))
        # print(os.path.join(img_path, "*", "*.jpg"))
        self.input_size = input_size
        if transforms is None:
            self.transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(input_size),
                                                              torchvision.transforms.ToTensor()])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # return sample = {"image": image, "label": label}
        try:
            img = PIL.Image.open(self.img_list[idx])
            img = self.transforms(img)
            # img = img[0:2,:,:]
        except OSError:
            print("OSError at combined_mask path ", self.img_list[idx])
            return None

        if self.img_list[idx].split('/')[-2] == "INVALID":
            label = 0
        # elif self.img_list[idx].split('/')[-2] == "VALID":
        else:
            label = 1
        sample = {"image": img, "label": label}
        return sample


def build_dataset_train(data_path, input_size):
    return BinaryClassifierDataset(os.path.join(data_path, "train"), input_size)


def build_dataset_eval(data_path, input_size):
    return BinaryClassifierDataset(os.path.join(data_path, "valid"), input_size)


def build_dataset_test(data_path, input_size):
    return BinaryClassifierDataset(os.path.join(data_path, "test"), input_size)
