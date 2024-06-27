import streamlit as st
import matplotlib.pyplot as plt    
import pickle as pkl
import torch
import cv2
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet18
from PIL import Image
import numpy as np


# Список файлов с данными
file_names = [
    'epoch_train_clf_loss.pkl', 'epoch_valid_clf_loss.pkl',
    'epoch_train_reg_loss.pkl', 'epoch_valid_reg_loss.pkl',
    'epoch_train_total_loss.pkl', 'epoch_valid_total_loss.pkl',
    'epoch_train_accuracy.pkl', 'epoch_valid_accuracy.pkl',
    'epoch_train_iou.pkl', 'epoch_valid_iou.pkl'
]
#Создаем словарь для хранения загруженных данных
hist_dict = {}
for file_name in file_names:
    with open(file_name, 'rb') as file:
        hist_dict[file_name.split('.')[0]] = pkl.load(file)

class LocModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        # фризим слои, обучать их не будем (хотя технически можно)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # задай классификационный блок
        self.clf = nn.Sequential(
            nn.Linear(512*8*8, 128),
            nn.Sigmoid(),
            nn.Linear(128, 3)
        )

        # задай регрессионный блок
        self.box = nn.Sequential(
            nn.Linear(512*8*8, 128),
            nn.Sigmoid(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, img):
        # задай прямой проход
        resnet_out = self.feature_extractor(img)
        resnet_out = resnet_out.view(resnet_out.size(0), -1)
        pred_classes = self.clf(resnet_out)
        pred_boxes = self.box(resnet_out)
        print(pred_classes.shape, pred_boxes.shape)
        return pred_classes, pred_boxes       
    

weights = 'weights_09.pt' 

my_model = LocModel()
my_model.load_state_dict(torch.load(weights,map_location=torch.device('cpu'))) 
my_model.eval()

valid_transforms = T.Compose(
    [
        T.Resize((227, 227)),
        T.ToTensor()])

def load_image(img): # загрузка изображения
    image = valid_transforms(img)        # применение трансформаций
    image = image.unsqueeze(0)      # добавление дополнительной размерности для батча
    return image

def predict(img):
    img = load_image(img)
    with torch.no_grad():
        pred_classes, pred_boxes = my_model(img)
        preds = torch.argmax(pred_classes, dim=1)
    return preds, pred_boxes

def draw_bounding_box(image, pred_box, color=(255, 0, 0)):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    pred_box = pred_box.squeeze().cpu().numpy()
    h, w = image.shape[:2]
    cv2.rectangle(image,
                  (int(pred_box[0] * w), int(pred_box[1] * h)),
                  (int(pred_box[2] * w), int(pred_box[3] * h)),
                  color, 2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

class_names = {0: 'cucumber',
            1: 'eggplant',
            2: 'mushroom'}

st.title('Модель локализации объектов ResNet18')
uploaded_file = st.file_uploader("Загрузите изображение (jpg или png)", type=["jpg", "png"])
if uploaded_file:
    image_open = Image.open(uploaded_file)
    predict_class, pred_box = predict(image_open)
    detected_image = draw_bounding_box(image_open, pred_box)
    pred_class_name = class_names[predict_class.item()]
    st.write(pred_class_name)
    st.image(detected_image)
else:
    st.write('Загрузите, пожалуйста, изображение')    

st.subheader("Метрики и графики модели")
# отображаем последние значения метрик
st.write(f'Accuracy train: {hist_dict["epoch_train_accuracy"][-1]:.4f}, Accuracy valid: {hist_dict["epoch_valid_accuracy"][-1]:.4f}')
st.write(f'Total Loss train: {hist_dict["epoch_train_total_loss"][-1]:.4f}, Total Loss valid: {hist_dict["epoch_valid_total_loss"][-1]:.4f}')
st.write(f'IOU train: {hist_dict["epoch_train_iou"][-1]:.4f}, IOU valid: {hist_dict["epoch_valid_iou"][-1]:.4f}')
# Функция рисования графиков
def plot_history(history, grid=True):
    fig, ax = plt.subplots(3, 2, figsize=(14, 20))
    fig.subplots_adjust(hspace=0.5)    
    ax[0, 0].plot(history['epoch_train_clf_loss'], label='train clf loss')
    ax[0, 0].plot(history['epoch_valid_clf_loss'], label='valid clf loss')
    ax[0, 0].set_title('Classification Loss')
    ax[0, 0].grid(grid)
    ax[0, 0].legend()    
    ax[0, 1].plot(history['epoch_train_reg_loss'], label='train reg loss')
    ax[0, 1].plot(history['epoch_valid_reg_loss'], label='valid reg loss')
    ax[0, 1].set_title('Regression Loss')
    ax[0, 1].grid(grid)
    ax[0, 1].legend()    
    ax[1, 0].plot(history['epoch_train_total_loss'], label='train total loss')
    ax[1, 0].plot(history['epoch_valid_total_loss'], label='valid total loss')
    ax[1, 0].set_title('Total Loss')
    ax[1, 0].grid(grid)
    ax[1, 0].legend()    
    ax[1, 1].plot(history['epoch_train_accuracy'], label='train accuracy')
    ax[1, 1].plot(history['epoch_valid_accuracy'], label='valid accuracy')
    ax[1, 1].set_title('Accuracy')
    ax[1, 1].grid(grid)
    ax[1, 1].legend()    
    ax[2, 0].plot(history['epoch_train_iou'], label='train IOU')
    ax[2, 0].plot(history['epoch_valid_iou'], label='valid IOU')
    ax[2, 0].set_title('IOU')
    ax[2, 0].grid(grid)
    ax[2, 0].legend()    
    return fig
# отображаем графики
st.pyplot(plot_history(hist_dict))