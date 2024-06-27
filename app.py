import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
import torchutils as tu
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from torchvision.models import resnet18
import streamlit as st
from PIL import Image



st.set_page_config(
    page_title='Локализация и детекция объектов')


st.sidebar.success('Выберите нужную страницу')

st.title('Локализация объектов и детекция объектов')