import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def PlotError(error,title):
    error = torch.div(error,torch.max(error))
    plt.figure()
    plt.title('Error plot')
    ax = plt.imshow(error.detach().numpy(),vmin = -1.1, vmax = 1.1,cmap ="YlGnBu")
    plt.savefig(title)

def PlotWeights(W1,W2,title1,title2,num_neuron):
    W1 = torch.reshape(W1,(num_neuron,28,50))
    W2 = torch.reshape(W2,(num_neuron,28,50))

    W1_RS = torch.zeros((280,500))
    W2_RS = torch.zeros((280,500))

    for i in range(10):
        for j in range(10):
            W1_RS[28*i:28*i+28,50*j:50*j+50] = W1[10*i+j,:,:]
            W2_RS[28*i:28*i+28,50*j:50*j+50] = W2[10*i+j,:,:]
    plt1 = plt.figure()
    plt.title('Initial Weights 1')
    plt.imshow(W1_RS,vmin = 0, vmax = 1, cmap = "hot_r")  
    plt1.savefig(title1)
    plt2 = plt.figure()
    plt.title('Initial Weights 2')
    plt.imshow(W2_RS, vmin = 0.0, vmax = 1, cmap = "hot_r")
    plt2.savefig(title2)

def ReconstructImage(spk2_recon,title,time):
    plt.figure()
    plt.imshow(spk2_recon.reshape((28,50)).detach().numpy(),vmin = 0, vmax= time, cmap = "binary")
    plt.savefig(title)

def PlotRaster(data,title):
    plt.figure(figsize=(10,6))
    img = plt.imshow(data,vmin = 0, vmax = data.max(), cmap = "hot_r",aspect = "auto")
    plt.colorbar(img,orientation = "horizontal")
    plt.savefig(title)

def PlotErrTime(err_rec, title):
    plt.figure()
    err= torch.stack(err_rec).detach().numpy()
    RollAvg = np.convolve(err,np.ones(20),'valid')/20
    img = plt.plot(np.linspace(1,err_rec.__len__()-19,err_rec.__len__()-19),RollAvg,'b-')
    plt.savefig(title)
    plt.close()
