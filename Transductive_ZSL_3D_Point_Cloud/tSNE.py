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
from src.datautil import *
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ModelNet', choices=['ModelNet', 'ScanObjectNN', 'McGill'], help='name of dataset i.e. ModelNet, ScanObjectNN, McGill')
parser.add_argument('--backbone', type=str, default='PointConv', choices=['EdgeConv', 'PointAugment', 'PointConv', 'PointNet', 'CurveNet'], help='name of backbone i.e. EdgeConv, PointAugment, PointConv, PointNet')
parser.add_argument('--config_path', type=str, required=True, help='configuration path')
args, unknown = parser.parse_known_args()

config_file = open(args.config_path, 'r')
config = yaml.load(config_file, Loader=yaml.FullLoader)

data_util = DataUtil(dataset=args.dataset, backbone=args.backbone, config=config)
data = data_util.get_data()

# Reading the data
d = pd.DataFrame(data["seen_feature_train"])
 
# save the labels into a variable l.
labels = pd.DataFrame(data['seen_labels_train'])

standardized_data = StandardScaler().fit_transform(d)
print(standardized_data.shape)

# Show first X data
# data = standardized_data[:1000, :]
# labels = labels[:1000]
 
# Show data corresponding to labels 1-10
labels_temp = data['seen_labels_train']
idx = [i for i, l in enumerate(labels_temp) if l in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
labels = labels.iloc[idx]
data = standardized_data[idx, :]

model = TSNE(n_components = 2, random_state = 0)
# configuring the parameters
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations
# for the optimization = 1000
 
tsne_data = model.fit_transform(data)
 
# creating a new data frame which
# help us in plotting the result data
tsne_data = np.vstack((tsne_data.T, labels.T)).T
tsne_df = pd.DataFrame(data = tsne_data,
     columns =("Dim_1", "Dim_2", "label"))
 
# Plotting the result of tsne
sn.scatterplot(data=tsne_df, x='Dim_1', y='Dim_2',
               hue='label', palette="bright")
plt.savefig("tSNE_CurveNet.png")
