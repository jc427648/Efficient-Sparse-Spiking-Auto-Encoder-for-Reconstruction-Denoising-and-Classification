import torch
import torchvision
import os
from PIL import Image
from torchvision.transforms import ToTensor

string = os.path.join(os.getcwd(),'FashionMNIST')

a = torchvision.datasets.FashionMNIST(string,train=True,download=True,transform= ToTensor) #All that is needed to download and train.
c = 1



