import os
import sys
import glob
import h5py
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.curvenet_cls import CurveNet512
from data import ModelNet40
import torch.nn as nn


# change this to your data root
DATA_DIR = 'CurveNet/data/'

def download_modelnet40():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        os.mkdir(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048'))
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data_cls(partition):
    download_modelnet40()
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40*hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def convert_data_to_ZSL():

    model_path = "CurveNet/core/convert/model512.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CurveNet512().to(device)
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



    # for idx, _ in enumerate(train_data):
    #     if train_labels[idx] in seen_index:
    #         train_data[idx] = train_data[idx].to(device)
    #         train_data[idx] = train_data[idx].permute(0, 2, 1)
    #         seen_train_data.append(model(train_data[idx]))
    #         seen_train_labels.append(train_labels[idx])
    #     else:
    #         train_data[idx] = train_data[idx].to(device)
    #         train_data[idx] = train_data[idx].permute(0, 2, 1)
    #         unseen_data.append(model(train_data[idx]))
    #         unseen_labels.append(train_labels[idx])

    # for idx, _ in enumerate(test_data):
    #     if test_labels[idx] in seen_index:
    #         test_data[idx] = test_data[idx].to(device)
    #         test_data[idx] = test_data[idx].permute(0, 2, 1)
    #         seen_test_data.append(model(test_data[idx]))
    #         seen_test_labels.append(test_labels[idx])
    #     else:
    #         unseen_data.append(model(test_data[idx]))
    #         test_data[idx] = test_data[idx].to(device)
    #         test_data[idx] = test_data[idx].permute(0, 2, 1)
    #         unseen_labels.append(test_labels[idx])

    return np.array(unseen_data), np.array(unseen_labels), np.array(seen_train_data), np.array(seen_train_labels), np.array(seen_test_data), np.array(seen_test_labels)



if __name__ == "__main__":
    
    unseen_data, unseen_labels, seen_train_data, seen_train_labels, seen_test_data, seen_test_labels = convert_data_to_ZSL()

    # unseen_data_tensor = torch.tensor(unseen_data)
    # unseen_labels_tensor = torch.tensor(unseen_labels)
    # seen_train_data_tensor = torch.tensor(seen_train_data)
    # seen_train_labels_tensor = torch.tensor(seen_train_labels)
    # seen_test_data_tensor = torch.tensor(seen_test_data)
    # seen_test_labels_tensor = torch.tensor(seen_test_labels)

    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/unseen_ModelNet10.mat', {'data': unseen_data})
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/unseen_ModelNet10_label.mat', {'label': unseen_labels})
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_train.mat', {'data': seen_train_data})
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_train_label.mat', {'label': seen_train_labels})
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_test.mat', {'data': seen_test_data})
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_test_label.mat', {'label': seen_test_labels})

    # savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/unseen.mat', {'data': unseen_data_tensor.detach().cpu().numpy()})
    # savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/unseen_label.mat', {'label': unseen_labels_tensor.detach().cpu().numpy()})
    # savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_train.mat', {'data': seen_train_data_tensor.detach().cpu().numpy()})
    # savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_train_label.mat', {'label': seen_train_labels_tensor.detach().cpu().numpy()})
    # savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_test.mat', {'data': seen_test_data_tensor.detach().cpu().numpy()})
    # savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_test_label.mat', {'label': seen_test_labels_tensor.detach().cpu().numpy()})
