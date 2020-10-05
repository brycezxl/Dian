import json

import torch
import torch.utils.data
from torch import optim
from torch.nn.functional import softmax
from tqdm import tqdm

from utils.utils import load_model


def test(data_loader, model, ckp_path):

    with torch.no_grad():
        optimizer = torch.optim.Adagrad(model.parameters())
        model, _, _, _, _ = load_model(model, optimizer, ckp_path)
        model.eval()

        threshold = 0.5
        outputs = []

        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            image = data["image"].cuda()
            name = data["name"]

            output = model(image)
            output = softmax(output, dim=-1)

            normal = output[:, 0]
            smoke = output[:, 1]
            call = output[:, 2]

            for j in range(image.size(0)):
                if smoke[j] > threshold / 2 and call[j] > threshold / 2:
                    category = "smoking_calling"
                    score = smoke[j] + call[j]
                elif call[j] > threshold:
                    category = "calling"
                    score = call[j]
                elif smoke[j] > threshold:
                    category = "smoking"
                    score = smoke[j]
                else:
                    category = "normal"
                    score = normal[j]

                name[j] = name[j].replace("jpg", "png")
                outputs.append({"image_name": name[j], "category": category, "score": "%.5f" % score})

    outputs.sort(key=lambda x: x['image_name'])
    with open("./log/result.json", "w+") as f:
        json.dump(outputs, f)
    print("Done.")

    return 0
