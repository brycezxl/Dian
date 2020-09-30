import os
from glob import glob

import PIL
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def picture_size():
    img_path = "../data/train/"
    img_list = glob(os.path.join(img_path, "*", "*.jpg"))
    width = 0
    height = 0
    for idx in range(len(img_list)):
        img = PIL.Image.open(img_list[idx])
        width += img.size[0]
        height += img.size[1]
    print(width / len(img_list), height / len(img_list))


if __name__ == '__main__':
    picture_size()
