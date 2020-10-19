import torchvision
import torch
import PIL
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import ImageFile
from glob import glob
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def generate_data(path_to_vehicle, path_to_lane, mode, input_size=(256, 512), transforms=None):
    haved_label_list = glob(os.path.join(root_dir, "combined_mask", mode, "*", "*.jpg"))
    vehicle_list = glob(os.path.join(path_to_vehicle, "*", "*.jpg"))
    print(len(haved_label_list), len(vehicle_list))
    for img in haved_label_list:
        img = os.path.join(path_to_vehicle, img.split("/")[-2],  img.split("/")[-1])
        if img in vehicle_list:
            vehicle_list.remove(img)
    print(len(haved_label_list), len(vehicle_list))

    if transforms == None:
        transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(input_size), torchvision.transforms.ToTensor()])
    else:
        transforms = transforms
    for idx, vehicle_path in tqdm(enumerate(vehicle_list)):
        try:
            vehicle = PIL.Image.open(vehicle_path)
            vehicle = transforms(vehicle)    
        except(OSError):
            # print("OSError at lane_mask path ", vehicle_path)
            continue

        try:
            lane_path = os.path.join(path_to_lane, vehicle_path.split("/")[-2], vehicle_path.split("/")[-1])
            lane = PIL.Image.open(lane_path)
            lane = transforms(lane)    
        except(OSError):
            # print("OSError at vehicle_mask path ", lane_path)
            continue

        vehicle = vehicle[0:1,:,:]
        combine = torch.cat((lane, vehicle), 0)
        combine = torch.cat((combine, torch.zeros(1, input_size[0], input_size[1])), 0)        
        img = np.array(np.transpose(combine, (1, 2, 0)), dtype=np.uint8) * 255
        save_path = os.path.join(root_dir, "combined_mask", mode, lane_path.split("/")[-2], lane_path.split("/")[-1])
        cv2.imwrite(save_path, img)
        # print(save_path)

def _generate_data(root_dir, input_size, mode):
    generate_data(os.path.join(root_dir, "vehicle_mask", mode), os.path.join(root_dir, "lane_mask", mode), mode, input_size)
    
if __name__ == "__main__":
    root_dir = "/home/rlee/cross_line_detection"
    test_dataset = _generate_data(root_dir, (256, 512), "train")