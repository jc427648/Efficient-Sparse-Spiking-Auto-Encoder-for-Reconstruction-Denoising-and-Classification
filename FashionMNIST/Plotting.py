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
    W1 = torch.reshape(W1,(num_neuron,28,28))
    W2 = torch.reshape(W2,(num_neuron,28,28))

    W1_RS = torch.zeros((280,280))
    W2_RS = torch.zeros((280,280))

    for i in range(10):
        for j in range(10):
            W1_RS[28*i:28*i+28,28*j:28*j+28] = W1[10*i+j,:,:]
            W2_RS[28*i:28*i+28,28*j:28*j+28] = W2[10*i+j,:,:]
    plt1 = plt.figure()
    plt.title('Initial Weights 1')
    plt.imshow(W1_RS,vmin = 0, vmax = 1, cmap = "hot_r")  
    plt1.savefig(title1)
    plt2 = plt.figure()
    plt.title('Initial Weights 2')
    plt.imshow(W2_RS, vmin = 0.0, vmax = 1, cmap = "hot_r")
    plt2.savefig(title2)
    plt.close()

def ReconstructImage(spk2_recon,title,time):
    plt.figure()
    plt.imshow(spk2_recon.reshape((28,28)).detach().numpy(),vmin = 0, vmax= time, cmap = "binary")
    plt.savefig(title)

def PlotRaster(data,title):
    plt.figure(figsize=(10,6))
    img = plt.imshow(data,vmin = 0, vmax = data.max(), cmap = "hot_r",aspect = "auto")
    plt.colorbar(img,orientation = "horizontal")
    plt.savefig(title)
    plt.close()

def PlotErrTime(err_rec, title):
    plt.figure()
    err= torch.stack(err_rec).detach().numpy()
    RollAvg = np.convolve(err,np.ones(10),'valid')/10
    img = plt.plot(np.linspace(1,err_rec.__len__()-9,err_rec.__len__()-9),RollAvg,'b-')
    plt.savefig(title)
    plt.close()

def PlotClusterOut(clust_data,title,num_class):
    plt.figure(figsize=(15,10))
    data = torch.zeros((10,num_class))
    for i in range(num_class):
        vals,counts = np.unique(clust_data[i],return_counts=True)
        for j in range(vals.shape[0]):

            data[vals[j],i] += counts[j]

    counts,AssignedLabel = data.max(0)
    ConfMatrix = torch.zeros((10,10))
    for i in range(num_class):
        ConfMatrix[:,AssignedLabel[i]] += data[:,i]

    TotAcc = 0
    for i in range(10):
        TotAcc += ConfMatrix[i,i]

    print('Total Accurate Spikes')
    print(TotAcc)

    img = sns.heatmap(
        data, annot=True, cmap="YlGnBu", cbar_kws={"label": "Scale"}
    )
    
    plt.xlabel("Output Neuron")
    plt.ylabel("Test Label")
    plt.savefig(title)
    plt.close()

    plt.figure(figsize = (15,10))
    img = sns.heatmap(
        ConfMatrix,annot= True, cmap = "YlGnBu",cbar_kws= {"label":"Scale"}
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.savefig('CM'+title)


def PlotClustWeight(weight,title,num_class):
    if num_class ==10:
        data = weight.reshape((num_class,28,28))
        data_RS = torch.zeros((28*5,28*2))
        for i in range(5):
            for j in range(2):
                data_RS[28*i:28*i+28,28*j:28*j+28] = torch.from_numpy(data[2*i+j,:,:])

    elif num_class == 25:
        data = weight.reshape((num_class,28,28))
        data_RS = torch.zeros((28*5,28*5))
        for i in range(5):
            for j in range(5):
                data_RS[28*i:28*i+28,28*j:28*j+28] = torch.from_numpy(data[5*i+j,:,:])

    elif num_class == 50:
        data = weight.reshape((num_class,28,28))
        data_RS = torch.zeros((28*10,28*5))
        for i in range(10):
            for j in range(5):
                data_RS[28*i:28*i+28,28*j:28*j+28] = torch.from_numpy(data[5*i+j,:,:])

    elif num_class == 100:
        data = weight.reshape((num_class,28,28))
        data_RS = torch.zeros((28*10,28*10))
        for i in range(10):
            for j in range(10):
                data_RS[28*i:28*i+28,28*j:28*j+28] = torch.from_numpy(data[10*i+j,:,:])

    else:
        data = weight.reshape((num_class,28,28))
        data_RS = torch.zeros((28*10,28*10))
        for i in range(10):
            for j in range(10):
                data_RS[28*i:28*i+28,28*j:28*j+28] = torch.from_numpy(data[10*i+j,:,:])



    
    plt.imshow(data_RS,vmin=0,vmax=1,cmap = "hot_r")
    plt.savefig(title)
    plt.close()

    

