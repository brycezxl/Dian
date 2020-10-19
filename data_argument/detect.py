import argparse

from PIL import Image

from models import *
from utils.argument import *
from utils.datasets import *
from utils.utils import *


# detection object, stored in /data/coco.names
OBJECT = 0  # car: 2, person: 0


def detect(
        cfg,
        data_cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    model.to(device).eval()
    dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])

    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred, _ = model(img)
        detections = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if detections is not None and len(detections) > 0:
            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            if (detections[:, -1] == OBJECT).sum() == 0:
                continue
            # create a new array to save car detections
            cars_detection = torch.zeros((detections[:, -1] == OBJECT).sum(), 7)
            count = 0

            # save all cars
            for c in range(detections.size(0)):
                if int(detections[c, -1]) == OBJECT:
                    cars_detection[count, :] = detections[c, :]
                    count += 1

            # select biggest car
            detections = select_car(cars_detection)

            # save img
            car = im0[int(detections[1]):int(detections[3]), int(detections[0]):int(detections[2]), :]
            im = Image.fromarray(car)
            save_path = save_path[:-4] + "_argument" + ".jpg"
            im.save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/home/zxl/traffic/fine-grained/data_argument/cfg/yolov3-spp.cfg',
                        help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='/home/zxl/traffic/fine-grained/data_argument/cfg/coco.data',
                        help='coco.data file path')
    parser.add_argument('--weights', type=str, default='/home/zxl/traffic/fine-grained/data_argument/weights/yolov3-spp.pt',
                        help='path to weights file')
    parser.add_argument('--images', type=str, default='/home/zxl/traffic/hualu_cup/data/train/calling_images',
                        help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )
