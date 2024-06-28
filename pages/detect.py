import streamlit as st
import torch
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms as T


st.title("Детекция объектов с YOLOv5")
uploaded_file = st.file_uploader("Upload image", type=["jpeg", "jpg", "png"])

@st.cache_resource
def get_model(conf):
    model = torch.hub.load(r'yolov5', 'custom', path=r'best.pt', source='local')
    model.eval()
    model.conf = conf
    print("Model loaded")
    return model


with st.sidebar:
    t = st.slider("Model conf", 0.0, 1.0, 0.1)

with st.spinner():
    model = get_model(t)

results = None
lcol, rcol = st.columns(2)
with lcol:
    if uploaded_file:
        img = Image.open(uploaded_file)
        results = model(img)
        st.image(img)

if results:
    with rcol:
        st.image(results.render())

