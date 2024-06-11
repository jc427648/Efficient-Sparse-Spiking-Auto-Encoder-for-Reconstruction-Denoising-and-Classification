import pickle
import torch
from Plotting import PlotError,PlotErrTime,PlotRaster,PlotWeights,ReconstructImage,PlotClusterOut,PlotClustWeight
import timeit
import os
import snntorch as snn

#This code trains for one epoch on a network that has already been trained.
#Would be good not to hard code these values just in case.

n_samples = 60000
log_interval = 1000
image_time = 200
pw = 10

# # traverse whole directory
# for root, dirs, files in os.walk(os.getcwd()):
#     # select file name
#     for file in files:
#         # check the extension of files
#         if file.endswith('.pckl'):
#             # print whole path of files
#             filestring = file

num_hidden = 5000
num_class = 100
num_epo = 5
decay1 = 5e-8
decay3 = 2.5e-6
sc1 = 0.999
max1 = 300
epo = 5

filename = 'NH%iNC%iEpo%iDc3_%.2eDc1_%.2eSc1_%.2eMax1%i.pckl' %(num_hidden,
                                                               num_class,
                                                               epo,
                                                               decay3,
                                                               decay1,
                                                               sc1,
                                                               max1)

f = open(filename,'rb')
network = pickle.load(f)
f.close()



num_hidden = 5000
epochs = num_epo
network.w3Update = True
network.params3[0] = 0.05 #Ap
network.params3[1] = 0.03 #An
network.params3[3] = 200 #Taun
# network.num_class = 50
# network.increase3 = network.decay3*200*network.num_class

PlotWeights(network.fc1.weight.data,
                            network.fc2.weight.data.transpose(0,1),
                            'W1 Final.png',
                            'W2 Final.png',
                            network.num_hidden)

PlotClustWeight(network.fc3.weight.detach().numpy(),
                                'Cluster weight Final.png',
                                network.num_class)

print('Lif1 Thresholds')
print(network.lif1.threshold.detach())
print('Lif2 Threshold')
print(network.lif2.threshold.detach())
print('Lif3 Thresholds')
print(network.lif3.threshold.detach())

with torch.no_grad():
    network.fc3.weight.copy_(torch.rand_like(network.fc3.weight.detach()))
    network.lif3.threshold.copy_(100*torch.ones_like(network.lif3.threshold.detach()))
    # network.lif3 = snn.Leaky(beta = 0.9,learn_threshold = True,threshold = 100*torch.ones(network.num_class),inhibition=True)



#Threshold wasn't randomised!!! That's why they're all in the same spot. Need to fix.

train_epochs = 2


assert n_samples >= 0 and n_samples <= 60000, "Invalid n_samples value."
print("Loading MNIST training samples...")
training_data = torch.load("train_images.pt")
training_labels = torch.load("train_labels.pt")
print('Increase1_%.2e_Decay1_%.2e' %(network.increase1,network.decay1))
print("Training...")
start_time = timeit.default_timer()
err_rec = []
clust_out = []
for i in range(network.num_class):
    clust_out.append([])
raster = torch.zeros((network.num_hidden, n_samples))

for T_epo in range(train_epochs):
    for idx in range(n_samples):
            network.HidActivity = torch.zeros(network.num_hidden)
            network.OutActivity = torch.zeros(network.num_inputs)
            network.ClustActivity = torch.zeros(network.num_class)
            network.spkFlag = False
            error,error_scalar,spk3_recon, spk2_recon, spk1_recon = network.PresentImage(training_data[idx],image_time,pw,UpdateParams = False)
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
                    1,
                    epochs,
                    timeit.default_timer()-start_time,
                    error_scalar
                    )
                )
                # print(network.lif2.threshold.detach())
                PlotWeights(network.fc1.weight.data,
                            network.fc2.weight.data.transpose(0,1),
                            'W1 %i.svg'%(idx+1),
                            'W2 %i.svg'%(idx+1),
                            network.num_hidden)
                title = 'TrRI_Imgs%i_Inc1%.2eDec1%.2eInc2%.2eDec2%.2ePW%i.png' %(idx+1,
                                                                                 network.increase1,
                                                                                 network.decay1,
                                                                                 network.increase2,
                                                                                 network.decay2,
                                                                                 pw)
                spk2_recon = (image_time-1)*training_data[idx] - error*(image_time-1)
                
                ReconstructImage(spk2_recon,title, image_time)
                # print(network.lif1.threshold.detach())
                # print(network.HidActivity)
                PlotClustWeight(network.fc3.weight.detach().numpy(),
                                'Peri_Cluster weight %iDc3_%.2e.png' %(idx+1,network.decay3),
                                network.num_class)
                # print(network.ClustActivity)
                # print(network.lif3.threshold.detach())

                # print(network.lif2.threshold.detach())
                #Periodic scaling
                with torch.no_grad():
                     network.lif3.threshold.mul_(0.9)
                
#Save network object.
f = open('Train_NH%iNC%iEpo%iDc3_%.2eDc1_%.2e.pckl' %(network.num_hidden,
                                                          network.num_class,
                                                          epochs+train_epochs,
                                                          network.decay3,
                                                          network.decay1), 'wb')
pickle.dump(network, f)
f.close()
#Plot all weights, raster plots and error w.r.t time.
PlotWeights(network.fc1.weight.data,
            network.fc2.weight.data.transpose(0,1),
            'W1_Inc1%.2eDec1%.2e.svg' %(network.increase1,
                                        network.decay1),
            'W2_Inc1%.2eDec1%.2e.svg' %(network.increase1,
                                        network.increase2
                                        ),
                                        network.num_hidden)
PlotRaster(raster.detach().numpy(),
           'Rast_Epo%i_Inc1%.2eDec1%.2eInc2%.2eDec2%.2eNH%i.svg' %(epochs+train_epochs,
                                                                   network.increase1,
                                                                   network.decay1,
                                                                   network.increase2,
                                                                   network.decay2,
                                                                   network.num_hidden))            
PlotErrTime(err_rec,
            'Train Error Time NH %i Epo %i.svg' %(network.num_hidden,
                                                          epochs+1))