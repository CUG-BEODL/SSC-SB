# SimCLR.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


# stage one ,unsupervised learning
class SimCLR(nn.Module):
    def __init__(self, input_dim, proj_size):
        super(SimCLR, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 256, bias=False),
                               nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True),
                               nn.Linear(256, proj_size, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
