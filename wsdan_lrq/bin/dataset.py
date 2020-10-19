import PIL
import torch
import sys
import torchvision
import os

import numpy as np ##del

import matplotlib.pyplot as plt
from PIL import ImageFile
from skimage import io
from glob import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True

class binary_classifier_dataset(torch.utils.data.Dataset):
    def __init__(self, path, input_size=(224, 224), transforms=None):
        self.img_list = glob(path + "*/*.jpg")
        if transforms == None:
            self.transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(input_size), torchvision.transforms.ToTensor()])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        #return sample = {"image": image, "label": label}
        sample = self.img_list[idx]
        
        try:
            image = PIL.Image.open(self.img_list[idx])
        except(OSError):
            print("OSError at image path ", self.img_list[idx])
        
        image = self.transforms(image)
        if self.img_list[idx].split('/')[6] == "VALID":
            label = 1
        elif self.img_list[idx].split('/')[6] == "INVALID":
            label = 0
        sample = {"image": image, "label": label}
        return sample

class cross_line_dataset(binary_classifier_dataset):
    def __init__(self, img_path, input_size=(256, 512), transforms=None):
        self.img_list = glob(os.path.join(img_path, "*", "*.jpg"))
        self.input_size = input_size
        if transforms == None:
            self.transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(input_size), torchvision.transforms.ToTensor()])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        #return sample = {"image": image, "label": label}        
        try:
            img = PIL.Image.open(self.img_list[idx])
            img = self.transforms(img)    
            img = img[0:2,:,:]
        except(OSError):
            print("OSError at combined_mask path ", self.img_list[idx])
            return None

        if self.img_list[idx].split('/')[-2] == "VALID":
            label = 1
        elif self.img_list[idx].split('/')[-2] == "INVALID":
            label = 0
        sample = {"image": img, "label": label}
        return sample

def build_dataset(root_dir, input_size, mode):
    return cross_line_dataset(os.path.join(root_dir, "combined_mask", mode), input_size)
    
if __name__ == "__main__":
    root_dir = "/home/rlee/cross_line_detection"
    test_dataset = build_dataset(root_dir, (256, 512), "train")
    batch_size = 8
    trainloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=True)
    for i ,data in enumerate(trainloader):
        if data == None:
            continue
        print(data["image"].size())
        """
        img = np.transpose(np.array(data["image"][0]), (1, 2, 0))
        plt.imshow(img)
        plt.show()
        """
