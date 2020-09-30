import json

import torch
import torch.utils.data
from torch import optim
from torch.nn.functional import softmax

from utils.utils import load_model


def test(data_loader, model, calling_ckp_path, smoking_ckp_path):

    with torch.no_grad():
        optimizer = torch.optim.Adagrad(model.parameters())
        model_calling, _, _, _, _ = load_model(model, optimizer, calling_ckp_path)
        model_smoking, _, _, _, _ = load_model(model, optimizer, smoking_ckp_path)
        model_calling.eval()
        model_smoking.eval()

        threshold = 0.5
        outputs = []

        for i, data in enumerate(data_loader):
            image = data["image"].cuda()
            name = data["name"]
            output_calling = softmax(model_calling(image)[0], dim=-1)
            output_smoking = softmax(model_smoking(image)[0], dim=-1)
            for j in range(image.size(0)):
                if output_calling[j, 1] > threshold and output_smoking[j, 1] > threshold:
                    category = "smoking_calling"
                    score = output_calling[j, 1] * output_smoking[j, 1]
                elif output_calling[j, 1] > threshold:
                    category = "calling"
                    score = output_calling[j, 1]
                elif output_smoking[j, 1] > threshold:
                    category = "smoking"
                    score = output_smoking[j, 1]
                else:
                    category = "normal"
                    score = output_calling[j, 0] * output_smoking[j, 0]

                outputs.append({"name": name[j], "category": category, "score": float(score)})

        with open("./log/result.json", "w+") as f:
            json.dump(outputs, f)
    return 0
