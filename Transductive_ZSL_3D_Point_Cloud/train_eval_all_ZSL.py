from __future__ import print_function
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import argparse
import scipy.io
import torch.optim as optim
from src.models import *
from src.models_blip import *
from src.models_clip import *
from src.loss import *
from src.util import *
from src.datautil import *
import yaml
import os
import csv
from datetime import datetime
import pandas as pd
import shutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

feature_dim = 1024

config_file = open("Transductive_ZSL_3D_Point_Cloud/config/ModelNet_config.yaml", 'r')
config = yaml.load(config_file, Loader=yaml.FullLoader)
dataset = 'ModelNet'

##### hyperparameters 
epoch = int(config['epoch'])
batch_size = int(config['batch_size'])
lr =  float(config['lr'])
amsgrad = True
eps = 1e-8
wd = float(config['wd'])

current_time = datetime.now()
current_time = current_time.strftime("%Y_%m_%d___%H_%M_%S")
log_path = "Logs/" + current_time + "/"
os.mkdir(log_path)
base_model_path = "Transductive_ZSL_3D_Point_Cloud/saved_model/CurveNet/model_CurveNet_ours_inductive.pth"
base_model_path = "Transductive_ZSL_3D_Point_Cloud/saved_model/"
backbones = ['CurveNet', 'CurveNet_BLIP', 'PointConv']
settings = ['inductive', 'transductive']
wordvec_methods = ['Word2Vec', 'BLIP', 'CLIP', 'CLIPDiminished']
tests = [] # backbone, setting, model_path_train, model_path_eval, wordvec_method


for backbone in backbones: # Create all needed tests
    backbone_copy = backbone
    if backbone == 'PointConv':
        backbone_copy = 'ModelNet'
    for setting in settings:
        if backbone is not 'CurveNet_BLIP':
            for wordvec_method in wordvec_methods:
                model_path_eval = base_model_path + backbone_copy + '/model_' + backbone + '_ours_' + setting + '_' + wordvec_method + '.pth'
                model_path_train = base_model_path + backbone_copy + '/'
                test = {'backbone': backbone, 'setting': setting, 'model_path_train': model_path_train, 'model_path_eval': model_path_eval, 'wordvec_method': wordvec_method}
                tests.append(test)
        else:
            wordvec_method = 'Word2Vec'
            model_path_eval = base_model_path + backbone_copy + '/model_' + backbone + '_ours_' + setting + '_' + wordvec_method + '.pth'
            model_path_train = base_model_path + backbone_copy + '/'
            test = {'backbone': backbone, 'setting': setting, 'model_path_train': model_path_train, 'model_path_eval': model_path_eval, 'wordvec_method': wordvec_method}
            tests.append(test)

shutil.copy("Transductive_ZSL_3D_Point_Cloud/config/ModelNet_config.yaml", log_path)
print(config)

results_training = []
results_training.append(['epochs', 'batch size', 'lr', 'wd'])
results_training.append([epoch, batch_size, lr, wd])
log_path_training = log_path + 'training/'
os.mkdir(log_path_training)
for test in tests: # train and log results

    str_tmp = [test['backbone'], test['setting'], test['wordvec_method']]
    results_training.append(str_tmp)

    if test['wordvec_method'] == 'Word2Vec' or test['wordvec_method'] == 'CLIPDiminished':
        model = S2F(feature_dim)
    elif test['wordvec_method'] == 'BLIP':
        model = S2F_BLIP(feature_dim)
    elif test['wordvec_method'] == 'CLIP':
        model = S2F_CLIP(feature_dim)

    model.to(device)
    if test['setting'] == 'inductive': # inductive
        model.apply(init_weights)
    else: # transductive
        path = test['model_path_train'] + 'model_' + test['backbone'] + '_ours_inductive_' + test['wordvec_method'] + '.pth'
        model.load_state_dict(torch.load(path))

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd, eps=eps, amsgrad=amsgrad)

    data_util = DataUtil(dataset=dataset, backbone=test['backbone'], config=config, wordvec_method=test['wordvec_method'])
    data = data_util.get_data()

    logfile_loop_path = log_path_training + test['backbone'] + "_" + test['setting'] + "_" + test['wordvec_method'] + '.csv'
    result_loop = []
    if test['setting'] == 'inductive': # inductive
        arr = np.arange(len(data['seen_labels_train']))
        step_batch_size = int(len(data['seen_labels_train'])/batch_size)-1

        for j in range(0,epoch):
            np.random.shuffle(arr)
            model.train()
            train_per_epoch_ours_inductive(model, optimizer, step_batch_size, arr, batch_size, data)
            result = calculate_accuracy_ours(model, data, config)
            result_loop.append([j+1, result['zsl_acc'], result['gzsl_seen'], result['gzsl_unseen'], result['gzsl_hm']])
            results_training.append([j+1, result['zsl_acc'], result['gzsl_seen'], result['gzsl_unseen'], result['gzsl_hm']])

            print("Epoch=", j+1, " ZSL: acc=", result['zsl_acc'],", GZSL: acc_S=",result['gzsl_seen'], ", acc_U=", result['gzsl_unseen'],", HM=",result['gzsl_hm'])

    else: # transductive
        unlabel_feature =  np.concatenate((data['seen_feature_test'], data['unseen_feature']), axis=0)
        arr = np.arange(len(data['seen_labels_train']))
        arr_unseen = np.arange(len(unlabel_feature))
        step_batch_size = int(len(data['seen_labels_train'])/batch_size)-1
        step_batch_size_unseen = int(len(data['unseen_labels'])/(batch_size/2))-1

        for j in range(0,epoch):
            np.random.shuffle(arr)
            model.train()
            train_per_epoch_ours_transductive(model, optimizer, step_batch_size, step_batch_size_unseen, arr, arr_unseen, batch_size, data, config)
            result = calculate_accuracy_ours(model, data, config)
            result_loop.append([j+1, result['zsl_acc'], result['gzsl_seen'], result['gzsl_unseen'], result['gzsl_hm']])
            results_training.append([j+1, result['zsl_acc'], result['gzsl_seen'], result['gzsl_unseen'], result['gzsl_hm']])

            print("Epoch=", j+1, " ZSL: acc=", result['zsl_acc'],", GZSL: acc_S=",result['gzsl_seen'], ", acc_U=", result['gzsl_unseen'],", HM=",result['gzsl_hm'])

    df = pd.DataFrame(result_loop)
    df = df.rename(columns={0: 'Epochs', 1: 'ZSL Acc', 2: 'GZSL Seen Acc', 3: 'GZSL Unseen Acc', 4: 'GZSL HM'})
    df.to_csv(logfile_loop_path)

    if not os.path.exists(test['model_path_train']):
        os.mkdir(test['model_path_train'])
    torch.save(model.state_dict(), test['model_path_eval'])

df = pd.DataFrame(results_training)
df = df.rename(columns={0: 'Epochs', 1: 'ZSL Acc', 2: 'GZSL Seen Acc', 3: 'GZSL Unseen Acc', 4: 'GZSL HM'})
df.to_csv(log_path + 'all_training.csv')




results_eval = []
results_eval.append(['epochs', 'batch size', 'lr', 'wd'])
results_eval.append([epoch, batch_size, lr, wd])
log_path_eval = log_path + 'eval/'
os.mkdir(log_path_eval)
for test in tests: # Evaluation

    logfile_loop_path = log_path_eval + test['backbone'] + "_" + test['setting'] + "_" + test['wordvec_method'] + '.csv'
    result_loop = []
    
    str_tmp = [test['backbone'], test['setting'], test['wordvec_method']]
    results_eval.append(str_tmp)

    if test['wordvec_method'] == 'Word2Vec' or test['wordvec_method'] == 'CLIPDiminished':
        model = S2F(feature_dim)
    elif test['wordvec_method'] == 'BLIP':
        model = S2F_BLIP(feature_dim)
    elif test['wordvec_method'] == 'CLIP':
        model = S2F_CLIP(feature_dim)
        
    model.to(device)
    model.load_state_dict(torch.load(test['model_path_eval']))

    data_util = DataUtil(dataset=dataset, backbone=test['backbone'], config=config, wordvec_method=test['wordvec_method'])
    data = data_util.get_data()

    result = calculate_accuracy_ours(model, data, config)
    result_loop.append([result['zsl_acc'], result['gzsl_seen'], result['gzsl_unseen'], result['gzsl_hm']])
    results_eval.append([result['zsl_acc'], result['gzsl_seen'], result['gzsl_unseen'], result['gzsl_hm']])

    print(" ZSL: acc=", result['zsl_acc'],", GZSL: acc_S=",result['gzsl_seen'], ", acc_U=", result['gzsl_unseen'],", HM=",result['gzsl_hm'])
    
    df = pd.DataFrame(result_loop)
    df = df.rename(columns={0: 'ZSL Acc', 1: 'GZSL Seen Acc', 2: 'GZSL Unseen Acc', 3: 'GZSL HM'})
    df.to_csv(logfile_loop_path)

df = pd.DataFrame(results_eval)
df = df.rename(columns={0: 'ZSL Acc', 1: 'GZSL Seen Acc', 2: 'GZSL Unseen Acc', 3: 'GZSL HM'})
df.to_csv(log_path + 'all_eval.csv')

saved_models_path = "Transductive_ZSL_3D_Point_Cloud/saved_model/"
shutil.copytree(saved_models_path, log_path + 'saved_model/')
