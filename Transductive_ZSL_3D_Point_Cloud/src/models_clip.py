import torch.nn as nn
import torch.nn.functional as F
from .curvenet_util import *

curve_config = {
        'default': [[100, 5], [100, 5], None, None],
        'long':  [[10, 30], None,  None,  None]
    }

# Ours (Inductive)
class S2F_CLIP(nn.Module):
    def __init__(self, feature_dim=1024):
        super(S2F_CLIP, self).__init__()
        self.fc3 = nn.Linear(768, feature_dim)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = self.relu(self.bn3(self.fc3(x)))
        return x

# Baseline (Inductive)
class F2S_CLIP(nn.Module):
    def __init__(self, feature_dim=1024):
        super(F2S_CLIP, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 768)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(768)

    def forward(self, x):
        x = self.tan(self.bn1(self.fc1(x)))
        return x


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)