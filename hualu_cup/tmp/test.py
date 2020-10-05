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

        outputs = []

        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):

            image = data["image"].cuda()
            name = data["name"]

            output = model(image)
            output = softmax(output, dim=-1)

            idx_to_name = ['normal', 'smoking', 'calling']

            for j in range(image.size(0)):
                if output[j, 1] > 0.4 and output[j, 2] > 0.4 and output[j, 0] < 0.2:
                    category = "smoking_calling"
                    score = output[j, 1] + output[j, 2]
                else:
                    idx = torch.argmax(output[j, :])
                    category = idx_to_name[idx]
                    score = output[j, idx]

                outputs.append({"category": category, "image_name": name[j], "score": round(float(score), 5)})

    outputs.sort(key=lambda x: int(x['image_name'].split('.')[0]))
    with open("./log/result.json", "w+") as f:
        f.write('[')
        for i in range(len(outputs)):
            f.write('{\"image_name\": \"' + outputs[i]["image_name"] +
                    '\", \"category\": \"' + outputs[i]["category"] +
                    '\", \"score\": ' + str(outputs[i]["score"]) + '}')
            if i != len(outputs) - 1:
                f.write(',\n')
            else:
                f.write(']')
    print("Done.")

    return 0
