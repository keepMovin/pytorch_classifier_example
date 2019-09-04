import torch
import torchvision
import torch.nn as nn
from torchvision import models

class NetWork(nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.model_ft = models.vgg16(pretrained=True)
        self.model_ft.classifier[6] = nn.Linear(4096, 3)


    def forward(self, x):
        classifier = self.model_ft(x)

        return classifier

