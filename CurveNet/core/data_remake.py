import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.curvenet_cls import CurveNet1024
from data import ModelNet40
import torch.nn as nn


def convert_data_to_ZSL():

    model_path = "CurveNet/core/convert/model_BLIP.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CurveNet1024().to(device)
    model.load_state_dict(torch.load(model_path))
    model = nn.DataParallel(model)

    seen_index   = np.int16([0,3,4,5,6,7,9,10,11,13,15,16,17,18,19,20,21,24,25,26,27,28,29,31,32,34,36,37,38,39])
    unseen_index = np.int16([1,2,8,12,14,22,23,30,33,35])

    train_loader = DataLoader(ModelNet40(partition='train', num_points=1024), num_workers=8,
                            batch_size=2, shuffle=False, drop_last=False)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=1024), num_workers=1,
                            batch_size=2, shuffle=False, drop_last=False)

    model.eval()

    unseen_data = []
    seen_train_data = []
    seen_test_data = []
    unseen_labels = []
    seen_train_labels = []
    seen_test_labels = []

    for data, label in train_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        for idx in range(logits.shape[0]):
            if label[idx].cpu().detach().numpy() in seen_index:
                seen_train_data.append(logits[idx].cpu().detach().numpy())
                label_trunc = np.where(seen_index == label[idx].cpu().detach().numpy())
                seen_train_labels.append(label_trunc[0][0])
            else:
                unseen_data.append(logits[idx].cpu().detach().numpy())
                label_trunc = np.where(unseen_index == label[idx].cpu().detach().numpy())
                unseen_labels.append(label_trunc[0][0])


    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        for idx in range(logits.shape[0]):
            if label[idx].cpu().detach().numpy() in seen_index:
                seen_test_data.append(logits[idx].cpu().detach().numpy())
                label_trunc = np.where(seen_index == label[idx].cpu().detach().numpy())
                seen_test_labels.append(label_trunc[0][0])
            else:
                unseen_data.append(logits[idx].cpu().detach().numpy())
                label_trunc = np.where(unseen_index == label[idx].cpu().detach().numpy())
                unseen_labels.append(label_trunc[0][0])

    return np.array(unseen_data), np.array(unseen_labels), np.array(seen_train_data), np.array(seen_train_labels), np.array(seen_test_data), np.array(seen_test_labels)



if __name__ == "__main__":
    
    unseen_data, unseen_labels, seen_train_data, seen_train_labels, seen_test_data, seen_test_labels = convert_data_to_ZSL()

    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet_BLIP/unseen_ModelNet10.mat', {'data': unseen_data}) # Change save data path to match
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet_BLIP/unseen_ModelNet10_label.mat', {'label': unseen_labels}) # Change save data path to match
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet_BLIP/seen_train.mat', {'data': seen_train_data}) # Change save data path to match
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet_BLIP/seen_train_label.mat', {'label': seen_train_labels}) # Change save data path to match
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet_BLIP/seen_test.mat', {'data': seen_test_data}) # Change save data path to match
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet_BLIP/seen_test_label.mat', {'label': seen_test_labels}) # Change save data path to match
