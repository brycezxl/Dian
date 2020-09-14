import cv2
import torch
import sys
import torchvision
import os
import PIL
from PIL import ImageFile

import numpy as np ##del

from glob import glob
ImageFile.LOAD_TRUNCATED_IMAGES = True

class binary_classifier_dataset(torch.utils.data.Dataset):
    def __init__(self, img_path, input_size=(32, 32), transforms=None):
        self.img_list = glob(os.path.join(img_path, "*", "*.jpg"))
        #print(os.path.join(img_path, "*", "*.jpg"))
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
            #img = img[0:2,:,:]
        except(OSError):
            print("OSError at combined_mask path ", self.img_list[idx])
            return None

        if self.img_list[idx].split('/')[-2] == "INVALID":
            label = 0
        elif self.img_list[idx].split('/')[-2] == "VALID":
            label = 1
        sample = {"image": img, "label": label}
        return sample

def build_dataset_train(data_path, input_size):
    return binary_classifier_dataset(os.path.join(data_path, "train"), input_size)
def build_dataset_eval(data_path, input_size):
    return binary_classifier_dataset(os.path.join(data_path, "valid"), input_size)
def build_dataset_test(data_path, input_size):
    return binary_classifier_dataset(os.path.join(data_path, "test"), input_size)

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
