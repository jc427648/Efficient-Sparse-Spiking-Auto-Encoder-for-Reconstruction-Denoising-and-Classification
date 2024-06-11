import pickle
import torch
from Plotting import PlotError,PlotErrTime,PlotRaster,PlotWeights,ReconstructImage,PlotClusterOut,PlotClustWeight
import timeit
import os
from FashionDL import getFashionMNIST
import snntorch as snn
import torch.nn as nn

#This code trains for one epoch on a network that has already been trained.
#Would be good not to hard code these values just in case.

n_samples = 60000
log_interval = 100
image_time = 300
pw = 10

# # traverse whole directory
# for root, dirs, files in os.walk(os.getcwd()):
#     # select file name
#     for file in files:
#         # check the extension of files
#         if file.endswith('.pckl'):
#             # print whole path of files
#             filestring = file

num_hidden = 10000
num_class = 1000
epochs_init = 3
increase1 = 5e-2
decay1 = 2.5e-8
increase3 = 1e0
decay3 = 1e-5
Ap1 = 5e-2
An1 = 2e-2
Ap3 = 5e-2
beta3 = 0.9

filename = 'NH%iNC%iEpo%iInc1_%.2eDc1_%.2eInc3_%.2eAp1_%.2eAn1_%.2eAp3_%.2e.pckl'%(
                                                                                        num_hidden,
                                                                                          num_class,
                                                                                          epochs_init,
                                                                                          increase1,
                                                                                          decay1,
                                                                                          increase3,
                                                                                          Ap1,
                                                                                          An1,
                                                                                          Ap3
                                                                                          )

f = open(filename,'rb')
network = pickle.load(f)
f.close()




epochs = 3
network.w3Update = True
#CHANGE UPDATE PARAMS BELOW TO ONLY TRAIN THIRD LAYER

network.params3[1] = 0.04 #An
network.params3[0] = 0.05 #Ap
network.params3[2] = 3
num_class = 2000
num_inputs = 784
network.decay3 = 2.5e-6
network.increase3 = 1e0
network.lif3Min = 1
network.lif3Max = 300

network.lif1Min = 1
network.lif1Max = 400
network.b = 0.15

lif3_scale = 0.99
lif1_scale = 0.999


network.num_class = num_class
network.lif3 = snn.Leaky(beta = beta3,learn_threshold = True,threshold = 100*torch.ones(num_class),inhibition=True)
network.fc3 = nn.Linear(num_inputs,num_class,bias = False)
network.mem3 = torch.zeros(num_class)
network.ClustActivity = torch.zeros(num_class)
network.spk3 = torch.zeros(num_class)
init_wt3 = torch.rand(num_class,num_inputs)
# with torch.no_grad():
#     network.fc3.weight.copy_(init_wt3)

# with torch.no_grad():
    # network.fc3.weight.copy_(torch.rand_like(network.fc3.weight.detach()))
    # network.lif3.threshold.copy_(100*torch.ones_like(network.lif3.threshold.detach()))
    # network.lif2.threshold.copy_(5*torch.ones_like(network.lif2.threshold.detach()))
    






assert n_samples >= 0 and n_samples <= 60000, "Invalid n_samples value."
print("Loading Fashion MNIST training samples...")
training_data,training_labels = getFashionMNIST(load_train=True,load_test=False,export_to_disk=False)
print('Increase1_%.2e_Decay1_%.2e' %(network.increase1,network.decay1))
print("Training...")
start_time = timeit.default_timer()
err_rec = []
clust_out = []
for i in range(network.num_class):
    clust_out.append([])
raster = torch.zeros((network.num_hidden, n_samples))


for epoch in range(epochs):
    training_data = training_data[torch.randperm(training_data.shape[0])]
    for idx in range(n_samples):
        network.HidActivity = torch.zeros(network.num_hidden)
        network.OutActivity = torch.zeros(network.num_inputs)
        network.ClustActivity = torch.zeros(network.num_class)
        network.spkFlag = False
        error,error_scalar,spk2_recon, spk3_recon, spk1_recon = network.PresentImage(training_data[idx],image_time,pw,UpdateParams = True)
        err_rec.append(error_scalar)
        raster[:,idx] = network.HidActivity

        if torch.sum(network.ClustActivity)>0:
            vals,indices = spk3_recon.min(0)
        else:
            vals,indices = network.mem3.max(1) #If no spike, then max membrane potential
        clust_out[indices].append(training_labels[idx].item())

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
                        'IT300W1 %i.png'%(idx+1),
                        'IT300W2 %i.png'%(idx+1),
                        network.num_hidden)
            title = 'IT300TrRI_Imgs%i_Inc1%.2eDec1%.2eInc2%.2eDec2%.2eNH%i.png' %(idx+1,
                                                                             network.increase1,
                                                                             network.decay1,
                                                                             network.increase2,
                                                                             network.decay2,
                                                                             network.num_hidden)
            ReconstructImage(spk2_recon,title, image_time)
            ReconstructImage(training_data[idx],'Input %i.png' %(idx+1),1)
            # print(network.lif1.threshold.detach())
            print('Hidden Activity')
            print(torch.sum(network.HidActivity))
            print('Reconstruction Activity')
            print(torch.sum(network.OutActivity))
            print('Cluster Activity')
            print(torch.sum(network.ClustActivity))
            # print(network.lif2.threshold.detach())
            PlotClustWeight(network.fc3.weight.detach().numpy(),
                            'IT300An%.2eDc3_%.2eNC%i Cluster weight %i.png' %(network.params3[1],
                                                                        network.decay3,
                                                                        num_class,
                                                                        idx+1),
                            network.num_class)
            # f = open('NH%iNC%iEpo%i.pckl' %(network.num_hidden,network.num_class,epoch+1), 'wb')
            # pickle.dump(network, f)
            # f.close()
            
            
            with torch.no_grad():
                     network.lif3.threshold.mul_(lif3_scale)
                     network.lif1.threshold.mul_(lif1_scale)
    f = open('IT300NH%iEpo%iNC%iDc3_%.2eAp%.2eAn%.2eTaup%.2eTaun%.2e.pckl' %(network.num_hidden,
                                                            epochs_init+epoch+1,
                                                            network.num_class,
                                                            network.decay3,
                                                            network.params3[0],
                                                            network.params3[1],
                                                            network.params3[2],
                                                            network.params3[3]), 'wb')
    pickle.dump(network, f)
    f.close()
PlotRaster(raster.detach().numpy(),'Rast_Epo%i_Inc1%.2eDec1%.2eInc2%.2eDec2%.2eNH%i.svg' %(epochs+1,
                                                                                           network.increase1,
                                                                                           network.decay1,
                                                                                           network.increase2,
                                                                                           network.decay2,
                                                                                           network.num_hidden))            
PlotErrTime(err_rec,'Train Error Time NH %i Epo %i.svg' %(network.num_hidden, epochs+1))