import pickle
import os
import tarfile
import numpy as np
import torch
from PIL import Image

def LoadCaltech():
    data = []    
    for i in range(2):
        if i == 0:
            for j in range(435):
                string = os.path.join(os.getcwd(),'Faces/image_%.4i.jpg'%(j+1))
                image = Image.open(string)
                image = image.convert("L")
                image = image.resize((50,28))
                image = 255 - np.asarray(image)
                data.append(np.asarray(image))
        elif i ==1:
            for j in range(798):
                string = os.path.join(os.getcwd(),'Motorbikes/image_%.4i.jpg'%(j+1))
                image = Image.open(string)
                image = image.convert("L")
                image = image.resize((50,28))
                data.append(np.asarray(image))

                                      
    #Need to downsample the image first, so that they are all the same size.
    #You then need to reshape all of the images after they're downsampled.
    data = np.array(data)
    data = torch.from_numpy(data)
    data = data/255

    return data

# filestring = os.path.join(os.getcwd(),'STDP-Autoencoder-Network\\101_ObjectCategories.tar.gz')
# file = tarfile.open(filestring)
# file.extractall('./Caltech_Data')
# file.close()

# a = LoadCaltech()