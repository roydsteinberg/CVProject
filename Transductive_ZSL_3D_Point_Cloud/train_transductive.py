from __future__ import print_function
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import argparse
import scipy.io
import torch.optim as optim
from src.models import *
from src.loss import *
from src.models import *
from src.util import *
from src.datautil import DataUtil
import yaml
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ModelNet', choices=['ModelNet', 'ScanObjectNN', 'McGill'], help='name of dataset i.e. ModelNet, ScanObjectNN, McGill')
parser.add_argument('--backbone', type=str, default='PointConv', choices=['EdgeConv', 'PointAugment', 'PointConv', 'PointNet', 'CurveNet'], help='name of backbone i.e. EdgeConv, PointAugment, PointConv, PointNet')
parser.add_argument('--method', type=str, default='ours', choices=['ours', 'baseline'], help='name of method i.e. ours, baseline')
parser.add_argument('--config_path', type=str, required=True, help='configuration path')
parser.add_argument('--model_path', type=str, required=True, help='model path')
args = parser.parse_args()

feature_dim = 2048 if (args.backbone == 'EdgeConv') else 1024

config_file = open(args.config_path, 'r')
config = yaml.load(config_file, Loader=yaml.FullLoader)
# print(config)

##### hyperparameters 
epoch = int(config['epoch'])
batch_size = int(config['batch_size'])
lr =  float(config['lr'])
amsgrad = True 
eps = 1e-8
wd = float(config['wd'])

model = S2F(feature_dim)
model.to(device)
path = args.model_path + 'model_' + args.backbone + '_' + args.method + '_inductive.pth'
model.load_state_dict(torch.load(path))
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd, eps=eps, amsgrad=amsgrad)

data_util = DataUtil(dataset=args.dataset, backbone=args.backbone, config=config)
data =data_util.get_data()
unlabel_feature =  np.concatenate((data['seen_feature_test'], data['unseen_feature']), axis=0)

arr = np.arange(len(data['seen_labels_train']))
arr_unseen = np.arange(len(unlabel_feature)) if args.method=='ours' else np.arange(len(data['unseen_labels']))
step_batch_size = int(len(data['seen_labels_train'])/batch_size)-1
step_batch_size_unseen = int(len(data['unseen_labels'])/(batch_size/2))-1


for j in range(0,epoch):
    np.random.shuffle(arr)
    model.train()
    if args.method == 'ours':
        train_per_epoch_ours_transductive(model, optimizer, step_batch_size, step_batch_size_unseen, arr, arr_unseen, batch_size, data, config)
    elif args.method == 'baseline':
        train_per_epoch_baseline_transductive(model, optimizer, step_batch_size, step_batch_size_unseen, arr, arr_unseen, batch_size, data, config)
    result = calculate_accuracy_ours(model, data, config)
    print("Epoch=", j+1, " ZSL: acc=", result['zsl_acc'],", GZSL: acc_S=",result['gzsl_seen'], ", acc_U=", result['gzsl_unseen'],", HM=",result['gzsl_hm'])

if not os.path.exists(args.model_path):
    os.mkdir(args.model_path)
path = args.model_path + 'model_' + args.backbone + '_' + args.method + '_transductive.pth'
torch.save(model.state_dict(), path)