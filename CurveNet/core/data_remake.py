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
from models.curvenet_cls import CurveNet
from data import ModelNet40


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

    seen_index   = np.int16([0,3,4,5,6,7,9,10,11,13,15,16,17,18,19,20,21,24,25,26,27,28,29,31,32,34,36,37,38,39])
    unseen_index = np.int16([1,2,8,12,14,22,23,30,33,35])

    train_data, train_labels = load_data_cls('train')
    test_data,  test_labels  = load_data_cls('test')

    unseen_data = []
    seen_train_data = []
    seen_test_data = []
    unseen_labels = []
    seen_train_labels = []
    seen_test_labels = []
    for idx, _ in enumerate(train_data):
        if train_labels[idx] in seen_index:
            seen_train_data.append(train_data[idx])
            seen_train_labels.append(train_labels[idx])
        else:
            unseen_data.append(train_data[idx])
            unseen_labels.append(train_labels[idx])

    for idx, _ in enumerate(test_data):
        if test_labels[idx] in seen_index:
            seen_test_data.append(test_data[idx])
            seen_test_labels.append(test_labels[idx])
        else:
            unseen_data.append(test_data[idx])
            unseen_labels.append(test_labels[idx])

    return unseen_data, unseen_labels, seen_train_data, seen_train_labels, seen_test_data, seen_test_labels



if __name__ == "__main__":
    
    unseen_data, unseen_labels, seen_train_data, seen_train_labels, seen_test_data, seen_test_labels = convert_data_to_ZSL()

    unseen_data_tensor = torch.tensor(unseen_data)[:, :1024].permute(0, 2, 1)
    unseen_labels_tensor = torch.tensor(unseen_labels)
    seen_train_data_tensor = torch.tensor(seen_train_data)[:, :1024].permute(0, 2, 1)
    seen_train_labels_tensor = torch.tensor(seen_train_labels)
    seen_test_data_tensor = torch.tensor(seen_test_data)[:, :1024].permute(0, 2, 1)
    seen_test_labels_tensor = torch.tensor(seen_test_labels)

    # savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/unseen_ModelNet10.mat', {'data': unseen_data_tensor[:, 0].detach().cpu().numpy()})
    # savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/unseen_ModelNet10_label.mat', {'label': unseen_labels_tensor.detach().cpu().numpy()})
    # savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_train.mat', {'data': seen_train_data_tensor[:, 0].detach().cpu().numpy()})
    # savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_train_label.mat', {'label': seen_train_labels_tensor.detach().cpu().numpy()})
    # savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_test.mat', {'data': seen_test_data_tensor[:, 0].detach().cpu().numpy()})
    # savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_test_label.mat', {'label': seen_test_labels_tensor.detach().cpu().numpy()})

    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/unseen.mat', {'data': unseen_data_tensor.detach().cpu().numpy()})
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/unseen_label.mat', {'label': unseen_labels_tensor.detach().cpu().numpy()})
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_train.mat', {'data': seen_train_data_tensor.detach().cpu().numpy()})
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_train_label.mat', {'label': seen_train_labels_tensor.detach().cpu().numpy()})
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_test.mat', {'data': seen_test_data_tensor.detach().cpu().numpy()})
    savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CurveNet/seen_test_label.mat', {'label': seen_test_labels_tensor.detach().cpu().numpy()})
