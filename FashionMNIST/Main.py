from Network import Net
from Plotting import PlotError, PlotWeights, ReconstructImage, PlotRaster,PlotErrTime,PlotClusterOut, PlotClustWeight
from FashionDL import getFashionMNIST
import sklearn
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import pandas as pd
import logging
import time
import timeit
import random
import os
import pickle



def Train(
        network,
        epochs = 2,
        n_samples = 60000,
        image_time = 200,
        pw = 10,
        log_interval = 1000,
        lif3_scale = 0.9,
        lif1_scale = 0.95
):
    assert n_samples >= 0 and n_samples <= 60000, "Invalid n_samples value."
    print("Loading FashionMNIST training samples...")
    training_data,training_labels = getFashionMNIST(load_train=True,
                                                    load_test=False,
                                                    export_to_disk=False)
    a = torch.randperm(60000)
    training_data = training_data[a]
    training_labels = training_labels[a]


    
    
    print("Training...")
    start_time = timeit.default_timer()
    err_rec = []
    clust_out = []
    for i in range(network.num_class):
        clust_out.append([])
    raster = torch.zeros((network.num_hidden, n_samples))
    recons = []
    for epoch in range(epochs):
        training_data = training_data[torch.randperm(training_data.shape[0])]
        for idx in range(n_samples):
            network.HidActivity = torch.zeros(network.num_hidden)
            network.OutActivity = torch.zeros(network.num_inputs)
            network.ClustActivity = torch.zeros(network.num_class)
            network.spkFlag = False
            error,error_scalar,spk2_recon, spk3_recon, spk1_recon = network.PresentImage(training_data[idx],
                                                                                         image_time,
                                                                                         pw,
                                                                                         UpdateParams = True)
            err_rec.append(error_scalar)
            raster[:,idx] = network.HidActivity

            if torch.sum(network.ClustActivity)>0:
                vals,indices = spk3_recon.min(0)
            else:
                vals,indices = network.mem3.max(1) #If no spike, then max membrane potential
            clust_out[indices].append(training_labels[idx].item())

            recons.append(spk2_recon)

            if (idx+1)% log_interval == 0:
                print(
                    "Training progress: sample (%.2e / %d) of epoch (%d / %d) - Elapsed time: %.4f. Current Error:%d" %(
                    idx+1,
                    n_samples,
                    epoch+1,
                    epochs,
                    timeit.default_timer()-start_time,
                    error_scalar
                    )
                )
                # print(network.lif2.threshold.detach())
                PlotWeights(network.fc1.weight.data, 
                            network.fc2.weight.data.transpose(0,1),
                            'W1 %i.png'%(idx+1),
                            'W2 %i.png'%(idx+1),
                            network.num_hidden)
                title = 'TrRI_Imgs%i_Inc1%.2eDec1%.2eInc2%.2eDec2%.2eNH%i.png' %(
                                                                                idx+1,
                                                                                network.increase1,
                                                                                 network.decay1,
                                                                                 network.increase2,
                                                                                 network.decay2,
                                                                                 network.num_hidden)
                ReconstructImage(spk2_recon,
                                 title, 
                                 image_time)
                ReconstructImage(training_data[idx],
                                 'Input %i.png' %(idx+1),
                                 1)
                # print(network.lif1.threshold.detach())
                print('Hidden Activity')
                print(network.HidActivity)
                # print(network.lif2.threshold.detach())
                PlotClustWeight(network.fc3.weight.detach().numpy(),
                                'Cluster weight %i.png' %(idx+1),
                                network.num_class)
                

                with torch.no_grad():
                     network.lif3.threshold.mul_(lif3_scale)
                     network.lif1.threshold.mul_(lif1_scale)
        f = open('NH%iNC%iEpo%iInc1_%.2eDc1_%.2eInc3_%.2eDc3_%.2eAp1_%.2eAn1_%.2eAp3_%.2e.pckl' %(
                                                        network.num_hidden,
                                                        network.num_class,
                                                        epoch+1,
                                                        network.increase1,
                                                        network.decay1,
                                                        network.increase3,
                                                        network.decay3,
                                                        network.params1[0],
                                                        network.params1[1],
                                                        network.params3[0]),
                        'wb')
        pickle.dump(network, f)
        f.close()


                     
    PlotRaster(raster.detach().numpy(),'RasterPlot.svg')
    PlotErrTime(err_rec,'Error Train Time.png')
    
    
    return network


def Test(
        network,
        epochs = 1,
        n_samples = 10000,
        image_time = 200,
        pw = 10,
        log_interval =500
        
):
    print("Loading Fashion MNIST test samples ...")
    test_data,test_labels = getFashionMNIST(load_train=False,load_test=True,export_to_disk=False)
    # test_data = test_data[torch.randperm(10000)]
    
    print("Testing...")
    start_time = timeit.default_timer()
    err_rec = []
    clust_out = []
    nonrecon = 0
    for i in range(network.num_class):
        clust_out.append([])

    for epoch in range(epochs):
        for idx in range(n_samples):
            network.HidActivity = torch.zeros(network.num_hidden)
            network.OutActivity = torch.zeros(network.num_inputs)
            network.ClustActivity = torch.zeros(network.num_class)
            error,error_scalar,spk2_recon,spk3_recon, spk1_recon = network.PresentImage(test_data[idx],image_time,pw,UpdateParams = False)
            err_rec.append(error_scalar)

            if torch.sum(network.OutActivity)==0:
                nonrecon +=1
            elif torch.sum(network.ClustActivity)>0:
                vals,indices = spk3_recon.min(0)
                clust_out[indices].append(test_labels[idx])
            else:
                vals,indices = network.mem3.max(1) #If no spike, then max membrane potential
                clust_out[indices].append(test_labels[idx])

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
                PlotError(torch.reshape(error,(28,28)),title=title)
                title = 'SBWD_Recon%iNH%iInc1_%.2eDec1_%.2e.png' %(idx+1,network.num_hidden,network.increase1,network.decay1)
                # spk2_recon = (image_time-1)*test_data[idx].flatten() - error*(image_time)
                ReconstructImage(spk2_recon,title, image_time)
                ReconstructImage(test_data[idx],'ReconInput%i.png' %(idx+1),1)
                # print(network.lif1.threshold.detach())
                # print(network.HidActivity)
                # print(network.lif2.threshold.detach())
                print('NonRecon')
                print(nonrecon)

                # print(network.rho)
    PlotErrTime(err_rec,'Error Test Time.png')

    PlotClusterOut(clust_data=clust_out,title = 'SBWD_Clustering Plot_Inc3%.2eDec3%.2eAp3%.2eAn3%.2eTauP3%.2eTauN3%.2e.svg' %(
        network.increase3,
        network.decay3,
        network.params3[0],
        network.params3[1],
        network.params3[2],
        network.params3[3]),num_class = network.num_class)
    f = open('store.pckl', 'wb')
    pickle.dump(network, f)
    f.close()
    return err_rec

