import torch.nn as nn
import torch.nn.functional as F
from .curvenet_util import *

curve_config = {
        'default': [[100, 5], [100, 5], None, None],
        'long':  [[10, 30], None,  None,  None]
    }

# Ours (Inductive)
class S2F(nn.Module):
    def __init__(self, feature_dim=1024):
        super(S2F, self).__init__()
        self.fc1 = nn.Linear(300, 512)
        self.fc2 = nn.Linear(512, 768)
        self.fc3 = nn.Linear(768, feature_dim)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(768)
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        return x

# Baseline (Inductive)
class F2S(nn.Module):
    def __init__(self, feature_dim=1024):
        super(F2S, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 768)
        self.fc2 = nn.Linear(768, 512)
        self.fc3 = nn.Linear(512, 300)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(768)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(300)

    def forward(self, x):
        x = self.tan(self.bn1(self.fc1(x)))
        x = self.tan(self.bn2(self.fc2(x)))
        x = self.tan(self.bn3(self.fc3(x)))
        return x
    
# CurveNet with last layers removed
class CurveNet(nn.Module):
    def __init__(self, num_classes=40, k=20, setting='default'):
        super(CurveNet, self).__init__()

        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=1024, radius=0.05, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][0])
        
        self.cic21 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=1024, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=256, radius=0.1, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=256, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=64, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=64, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][3])

        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

    def forward(self, xyz):
        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)
 
        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)
        
        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        return x



def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)