import torch
import torch.nn as nn
from torchvision.models import resnet50

# DEVICE = 'cpu'

class ResNet50Custom(nn.Module):
    def __init__(self):
        super(ResNet50Custom, self).__init__()
        # Создание модели ResNet50 без предобученных весов
        self.model = resnet50(pretrained=False)
      #  self.model.to(DEVICE)
        # Замена последнего полносвязного слоя
        self.model.fc = nn.Linear(2048, 6)
    def forward(self, x):
        return self.model(x)
