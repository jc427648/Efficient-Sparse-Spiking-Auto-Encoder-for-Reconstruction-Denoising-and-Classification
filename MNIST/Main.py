from Network import Net
from Plotting import PlotError, PlotWeights, ReconstructImage, PlotRaster,PlotErrTime,PlotClusterOut, PlotClustWeight
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


# @profile
def Train(
        network,
        epochs = 1,
        n_samples = 60000,
        image_time = 200,
        pw = 10,
        log_interval = 500,
        lif3_scale = 0.9,
        lif1_scale = 0.999
):
    assert n_samples >= 0 and n_samples <= 60000, "Invalid n_samples value."
    print("Loading MNIST training samples...")
    training_data = torch.load("train_images.pt")
    training_labels = torch.load("train_labels.pt")
    
    print("Training...")
    start_time = timeit.default_timer()
    err_rec = []
    clust_out = []
    for i in range(network.num_class):
        clust_out.append([])
    raster = torch.zeros((network.num_hidden, n_samples))
    print('Lif1Scale')
    print(lif1_scale)
    HidSum = 0
    OutSum = 0
    ClustSum = 0
    Total_W1 = 0
    Total_W2 = 0
    Total_W3 = 0
    for epoch in range(epochs):
        for idx in range(n_samples):
            network.HidActivity = torch.zeros(network.num_hidden)
            network.OutActivity = torch.zeros(network.num_inputs)
            network.ClustActivity = torch.zeros(network.num_class)
            network.spkFlag = False
            error,error_scalar,spk3_recon, spk2_recon, spk1_recon, Syn1_count, Syn2_count, Syn3_count  = network.PresentImage(training_data[idx],image_time,pw,UpdateParams = True)
            err_rec.append(error_scalar)
            raster[:,idx] = network.HidActivity

            if torch.sum(network.ClustActivity)>0:
                vals,indices = spk3_recon.min(0)
            else:
                vals,indices = network.mem3.max(1) #If no spike, then max membrane potential
            clust_out[indices].append(training_labels[idx].item())

            HidSum = HidSum + torch.sum(network.HidActivity)
            OutSum = OutSum + torch.sum(network.OutActivity)
            ClustSum = ClustSum + torch.sum(network.ClustActivity)
            Total_W1 += Syn1_count
            Total_W2 += Syn2_count
            Total_W3 += Syn3_count

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
                PlotWeights(network.fc1.weight.data, 
                            network.fc2.weight.data.transpose(0,1),
                            'W1 %iSc1_%.2eMax1_%i.png'%(idx+1,lif1_scale,network.lif1Max),
                            'W2 %i.png'%(idx+1),
                            network.num_hidden)
                title = 'TrRI_Imgs%i_Inc1%.2eDec1%.2eInc2%.2eDec2%.2ePW%iSc1%.2eMax1_%i.svg' %(idx+1,
                                                                                 network.increase1,
                                                                                 network.decay1,
                                                                                 network.increase2,
                                                                                 network.decay2,
                                                                                 pw,
                                                                                 lif1_scale,
                                                                                 network.lif1Max)
                spk2_recon = (image_time-1)*training_data[idx] - error*(image_time-1)

                
                ReconstructImage(spk2_recon,title, image_time)
                # print(network.lif1.threshold.detach())
                # print(network.HidActivity)
                PlotClustWeight(network.fc3.weight.detach().numpy(),
                                'Cluster weight %iInc1_%.2eSc1_%.2eMax1_%i.png' %(idx+1,
                                                                        network.increase1,
                                                                         lif1_scale,
                                                                         network.lif1Max),
                                network.num_class)
                # print(network.ClustActivity)
                # print(network.lif3.threshold.detach())

                # print(network.lif2.threshold.detach())
                #Scale network thresholds
                with torch.no_grad():
                     network.lif3.threshold.mul_(lif3_scale)
                     network.lif1.threshold.mul_(lif1_scale)
        f = open('Inc1_%.2eNH%iNC%iEpo%iDc3_%.2eDc1_%.2eSc1_%.2eMax1%i.pckl' %(
                                                            network.increase1,
                                                            network.num_hidden,
                                                            network.num_class,
                                                            epoch+1,
                                                            network.decay3,
                                                            network.decay1,
                                                            lif1_scale,
                                                            network.lif1Max), 'wb')
        pickle.dump(network, f)
        f.close()

    PlotRaster(raster.detach().numpy(),
               'RasterPlot.svg')
    PlotErrTime(err_rec,
                'Error Train Time.png')
    PlotClusterOut(clust_data=clust_out,
                   title = 'Clustering Plot Train.svg', 
                   num_class=network.num_class)
    PlotClustWeight(network.fc3.weight.detach().numpy(),
                    'Cluster Final weight%i.svg'%(network.num_class),
                    network.num_class)
    print('Hidden Train Activity')
    print(HidSum)
    print('Out Activity Train')
    print(OutSum)
    print('Cluster activity Train')
    print(ClustSum)
    print('Increase 1')
    print(network.increase1)
    print('Total W1 Updates')
    print(Total_W1)
    print('Total W2 Updates')
    print(Total_W2)
    print('Total W3 Updates')
    print(Total_W3)
    
    return network


def Test(
        network,
        epochs = 1,
        n_samples = 10000,
        image_time = 1000,
        pw = 10,
        log_interval =100
        
):
    print("Loading MNIST test samples ...")
    test_data = torch.load('test_images.pt')
    test_labels = torch.load('test_labels.pt')
    
    print("Testing...")
    start_time = timeit.default_timer()
    err_rec = []
    clust_out = []
    nonrecon = 0
    #Below are just for debugging
    in_clust = []
    non_in = 0

    for i in range(network.num_class):
        clust_out.append([])
    for i in range(network.num_hidden):
        in_clust.append([])

    HidSum = 0
    OutSum = 0
    ClustSum = 0
    Total_W1 = 0
    Total_W2 = 0
    Total_W3 = 0

    for epoch in range(epochs):
        for idx in range(n_samples):
            network.HidActivity = torch.zeros(network.num_hidden)
            network.OutActivity = torch.zeros(network.num_inputs)
            network.ClustActivity = torch.zeros(network.num_class)
            error,error_scalar, spk3_recon, spk2_recon, spk1_recon, Syn1_count, Syn2_count, Syn3_count  = network.PresentImage(test_data[idx],image_time,pw,UpdateParams = False)
            Total_W1 += Syn1_count
            Total_W2 += Syn2_count
            Total_W3 += Syn3_count
            err_rec.append(error_scalar)

            if torch.sum(network.OutActivity)==0:
                nonrecon +=1
            elif torch.sum(network.ClustActivity)>0:
                vals,indices = spk3_recon.min(0)
                # print('Spk3_recon')
                # print(spk3_recon)
                clust_out[indices].append(test_labels[idx])
            else:
                vals,indices = network.mem3.max(1) #If no spike, then max membrane potential
                clust_out[indices].append(test_labels[idx])

            #Debugging for first layer (see if it classifies well.)
            if torch.sum(network.HidActivity)==0:
                non_in +=1
            else:
                vals, indices = spk1_recon.min(0)
                in_clust[indices].append(test_labels[idx])



            HidSum = HidSum + torch.sum(network.HidActivity)
            OutSum = OutSum + torch.sum(network.OutActivity)
            ClustSum = ClustSum + torch.sum(network.ClustActivity)

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
                title = 'TeRI%iInc1_%.2e.svg' %(idx+1,network.increase1)
                spk2_recon = (image_time-1)*test_data[idx] - error*(image_time)
                
                ReconstructImage(spk2_recon,title, image_time)
                ReconstructImage(test_data[idx],'Input %i.svg' %(idx+1),1)
                # print(network.lif1.threshold.detach())
                print('Hidden Layer Activity')
                print(torch.sum(network.HidActivity))
                print('Recontruction Layer Activity')
                print(torch.sum(network.OutActivity))
                print('Cluster Layer Activity')
                print(torch.sum(network.ClustActivity))
                # print(network.lif3.threshold.detach())
                print('Non Recon')
                print(nonrecon)
                

                # print(network.rho)
    PlotErrTime(err_rec,'Error Test Time.svg')
    #Assign labels to maximum counts.

    


    PlotClusterOut(clust_data=clust_out,
                   title = 'Clustering Plot_Inc3%.2eDec3%.2eAp3%.2eAn3%.2eTauP3%.2eTauN3%.2eNC%i.svg' %(
                            network.increase3,
                            network.decay3,
                            network.params3[0],
                            network.params3[1],
                            network.params3[2],
                            network.params3[3],
                            network.num_class),
                    num_class = network.num_class)
    # print('Debugging')
    # PlotClusterOut(clust_data = in_clust, title = 'Debugging.png',num_class = network.num_hidden)
    # f = open('NH%iNC%i.pckl'%(network.num_hidden,network.num_class), 'wb')
    # pickle.dump(network, f)
    # f.close()
    print('Increase 1')
    print(network.increase1)
    print('Hidden Test Activity')
    print(HidSum)
    print('Out Activity Test')
    print(OutSum)
    print('Cluster activity Test')
    print(ClustSum)
    print('Total W1 Updates')
    print(Total_W1)
    print('Total W2 Updates')
    print(Total_W2)
    print('Total W3 Updates')
    print(Total_W3)
    return err_rec