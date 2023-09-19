import os
import torch
import torch.nn as nn
from models.curvenet_cls import CurveNet
from models.curvenet_cls import CurveNet512
from torchinfo import summary

def convert_model(model_path, save_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dict = torch.load(model_path)

    del model_dict['module.conv2.bias']
    del model_dict['module.conv2.weight']

    model = CurveNet512().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(model_dict)

    if not os.path.exists(save_path):
        os.mkdir(save_path[:-9])
    torch.save(model.module.state_dict(), save_path)
    
def print_model(model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CurveNet512().to(device)
    model.load_state_dict(torch.load(model_path))

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


if __name__ == "__main__":

    model_path = "CurveNet/pretrained/cls/models/model.t7"
    save_path = "CurveNet/core/convert/model512.pth"

    convert_model(model_path, save_path)

    print_model(save_path)
