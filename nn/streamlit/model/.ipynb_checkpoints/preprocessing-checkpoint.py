import torch
from torchvision import transforms as T

test_transforms = T.Compose([
    T.Resize((75, 75)),
    T.ToTensor(),
    T.Normalize((0.43, 0.46, 0.45), (0.27, 0.27, 0.3))
])

def preprocess(img):
    return test_transforms(img)