import torch
import torchvision
import os
from PIL import Image
from torchvision.transforms import ToTensor




def getFashionMNIST(
        load_train = True,
        load_test = False,
        export_to_disk = True
):
    string = os.path.join(os.getcwd(),'FashionMNIST')
    if load_train ==True:
        dataset = torchvision.datasets.FashionMNIST(string,train=True,download=True,transform= ToTensor)
        data = dataset.data

        data = 1-data/255 #TTFS
        data = data.reshape((60000,784))
        labels = dataset.targets
    elif load_test ==True:
        dataset = torchvision.datasets.FashionMNIST(string,train = False,download =True,transform=ToTensor)
        data = dataset.data

        data = 1-data/255 #TTFS
        data = data.reshape((10000,784))
        labels = dataset.targets
        
    
    
    return data,labels


        