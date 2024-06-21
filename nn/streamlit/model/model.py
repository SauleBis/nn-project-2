import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class myResNet_50(nn.Module):
    def __init__(self):
         super().__init__()
         self.model = resnet50(pretrained=False)
         self.model.fc = nn.Linear(2048, 200)
         # замораживаем слои
         for i in self.model.parameters():
             i.requires_grad = False
        # размораживаем только последний, который будем обучать
         self.model.fc.weight.requires_grad = True
         self.model.fc.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)
