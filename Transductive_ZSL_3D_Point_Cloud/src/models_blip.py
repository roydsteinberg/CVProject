import torch.nn as nn
import torch.nn.functional as F
from .curvenet_util import *

curve_config = {
        'default': [[100, 5], [100, 5], None, None],
        'long':  [[10, 30], None,  None,  None]
    }

# Ours (Inductive)
class S2F_BLIP(nn.Module):
    def __init__(self, feature_dim=1024):
        super(S2F_BLIP, self).__init__()
        self.fc1 = nn.Linear(252, 458)
        self.fc2 = nn.Linear(458, 768)
        self.fc3 = nn.Linear(768, feature_dim)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(458)
        self.bn2 = nn.BatchNorm1d(768)
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        return x

# Baseline (Inductive)
class F2S_BLIP(nn.Module):
    def __init__(self, feature_dim=1024):
        super(F2S_BLIP, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 768)
        self.fc2 = nn.Linear(768, 458)
        self.fc3 = nn.Linear(458, 252)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(768)
        self.bn2 = nn.BatchNorm1d(458)
        self.bn3 = nn.BatchNorm1d(252)

    def forward(self, x):
        x = self.tan(self.bn1(self.fc1(x)))
        x = self.tan(self.bn2(self.fc2(x)))
        x = self.tan(self.bn3(self.fc3(x)))
        return x


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)