import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from torchvision import datasets
from einops import rearrange
from einops.layers.torch import Rearrange
import sys
from math import sqrt
from tqdm.auto import tqdm
import pandas as pd
import cv2
from PIL import Image

test_df = pd.read_csv("C:/test/image_info.csv")
ans = list(test_df["label"][:])

model = torch.jit.load('./model2.pth')
model.eval() # 進入評估狀態

device = torch.device("cuda")

label = {'asparagus': 0, 'bambooshoots': 1, 'betel': 2, 'broccoli': 3, 'cauliflower': 4, 'chinesecabbage': 5, 'chinesechives': 6, 'custardapple': 7, 'grape': 8, 'greenhouse': 9, 'greenonion': 10, 'kale': 11, 'lemon': 12, 'lettuce': 13, 'litchi': 14, 'longan': 15, 'loofah': 16, 'mango': 17, 'onion': 18, 'others': 19, 'papaya': 20, 'passionfruit': 21, 'pear': 22, 'pennisetum': 23, 'redbeans': 24, 'roseapple': 25, 'sesbania': 26, 'soybeans': 27, 'sunhemp': 28, 'sweetpotato': 29, 'taro': 30, 'tea': 31, 'waterbamboo': 32}
transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

model = model.to(device)
torch.no_grad()
ac = 0
for i in range(100):
    label_num = 0
    img = Image.open("C:/pre/asparagus-001/" + test_df["Img"][i])
    image = transform_test(img).unsqueeze(0)
    img_ = image.to(device)
    outputs = model(img_)
    _, predicted = torch.max(outputs, 1)
    if(predicted[0].tolist()==label_num):
        ac += 1
    print(predicted[0].tolist(), label_num)


# print(predicted)
print(ac/100)

