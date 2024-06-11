from Network import Net
from Plotting import PlotError, PlotWeights, ReconstructImage, PlotRaster,PlotErrTime
from CaltechDL import LoadCaltech
# import sklearn
# from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import pandas as pd
# import logging
# import time
import timeit
# import random
# import os
import pickle


# @profile
def Train(
        network,
        training_data,
        epochs = 1,
        n_samples = 1193,
        image_time = 200,
        pw = 1,
        log_interval = 500
):
    assert n_samples >= 0 and n_samples <= 1233, "Invalid n_samples value."
    # print("Loading Caltech training samples...")
    # training_data = LoadCaltech()
    # training_data = training_data[torch.randperm(1233)]
    
    print("Training...")
    start_time = timeit.default_timer()
    err_rec = []
    raster = torch.zeros((network.num_hidden, n_samples))
    for epoch in range(epochs):
        training_data = training_data[torch.randperm(training_data.shape[0])]
        for idx in range(n_samples):
            network.HidActivity = torch.zeros(network.num_hidden)
            network.OutActivity = torch.zeros(network.num_inputs)
            error,error_scalar = network.PresentImage(training_data[idx],image_time,pw,UpdateParams = True)
            err_rec.append(error_scalar)
            raster[:,idx] = network.HidActivity

            if (idx+1)% log_interval == 0:
                print(
                    "Training progress: sample (%.2e / %d) of epoch (%d / %d) - Elapsed time: %.4f. Current Error:%d" %(
                    idx+1,
                    n_samples,
                    epoch,
                    epochs,
                    timeit.default_timer()-start_time,
                    error_scalar
                    )
                )
                # print(network.lif2.threshold.detach())
                PlotWeights(network.fc1.weight.data, network.fc2.weight.data.transpose(0,1),'W1 %i Epo%i.png'%(idx+1,epoch+1),'W2 %i Epo%i.png'%(idx+1,epoch+1),network.num_hidden)
                title = 'TrRI_Imgs%i_Inc1%.2eDec1%.2eAP1%.2eTP1%.2eEpo%iNH%i.png' %(idx+1,network.increase1,network.decay1,network.params1[0],network.params1[2],epoch+1,network.num_hidden)
                spk2_recon = (image_time-1)*training_data[idx].flatten() - error*(image_time-1)
                ReconstructImage(spk2_recon,title, image_time)
                print(network.lif1.threshold.detach())
                print(torch.sum(network.HidActivity))
                # print(network.lif2.threshold.detach())
    PlotRaster(raster.detach().numpy(),'RasterPlot.svg')
    PlotErrTime(err_rec,'Error Train Time.png')
    return network


def Test(
        network,
        test_data,
        epochs = 1,
        n_samples = 40,
        image_time = 200,
        pw = 1,
        log_interval =1        
):
    print("Loading Caltech test samples ...")
    # test_data = LoadCaltech()
    # test_data = test_data[torch.randperm(1233)]
    
    print("Testing...")
    start_time = timeit.default_timer()
    err_rec = []
    for epoch in range(epochs):
        for idx in range(n_samples):
            network.HidActivity = torch.zeros(network.num_hidden)
            network.OutActivity = torch.zeros(network.num_inputs)
            error,error_scalar = network.PresentImage(test_data[idx],image_time,pw,UpdateParams = False)
            err_rec.append(error_scalar)

            if (idx+1)% log_interval == 0:
                print(
                    "Testing progress: sample (%d / %d) of epoch (%d / %d) - Elapsed time: %.4f. Current Error:%d" %(
                    idx+1,
                    n_samples,
                    epoch + 1,
                    epochs,
                    timeit.default_timer()-start_time,
                    err_rec[idx]
                    )
                )
                title = '%i.png' %(idx+1)
                PlotError(torch.reshape(error,(28,50)),title=title)
                title = 'Reconstruct %i.svg' %(idx+1)
                spk2_recon = (image_time-1)*test_data[idx].flatten() - error*(image_time)
                ReconstructImage(spk2_recon,title, image_time)
                ReconstructImage(test_data[idx],'Input %i.svg' %(idx+1),1)
                # print(network.lif1.threshold.detach())
                print('Hidden Activity')
                print(torch.sum(network.HidActivity))

                # print(network.rho)
    PlotErrTime(err_rec,'Error Test Time.png')
    f = open('store.pckl', 'wb')
    pickle.dump(network, f)
    f.close()
    return err_rec

