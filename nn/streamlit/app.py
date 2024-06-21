import streamlit as st
import requests
import torch
from PIL import Image
from io import BytesIO
from model.model import myResNet_50
from model.preprocessing import preproccessing

  # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource()                         
def load_model():
   model = myResNet_50()
   #model.to(DEVICE)
   model.load_state_dict(torch.load('model/weights_08.pt')) 
   #model.to(DEVICE)
   model.eval()
   return model

model = load_model()
#model.to(DEVICE)

def predict(img):
    img = preproccessing(img)
    pred = model(img)
    return pred

st.title("Нейронные сети: классификация изображений")

tab1, tab2 = st.tabs(["По ссылке", "С локального диска"])

with tab1:
    # Загрузка изображения по ссылке
    url = st.text_input("Введите URL изображения:")
    if url:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption='Загруженное изображение по ссылке', use_column_width=True)
            prediction = predict(img)
            st.write(prediction)
        except Exception as e:
            st.error(f"Не удалось загрузить изображение по ссылке. Ошибка: {e}")
with tab2:
    # Загрузка нескольких изображений с локального диска
    uploaded_files = st.file_uploader("Выберите изображения", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption=f'Загруженное изображение: {uploaded_file.name}', use_column_width=True)
                prediction = predict(uploaded_file)
                st.write(prediction)

            except Exception as e:
                st.error(f"Не удалось загрузить изображение {uploaded_file.name}. Ошибка: {e}")
 
                         
                         



