import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_features, init_weight=True):
        super(SpatialAttention, self).__init__()
        self.layer1 = nn.Sequential(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        self.layer1.apply(self.weights_init)
        self.layer2.apply(self.weights_init)
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    def forward(self, x):
        spatial_avg_pooling = torch.mean(x, dim=1, keepdim=True)
        out = self.layer1(spatial_avg_pooling) + x
        out = self.layer2(out)
        return out
