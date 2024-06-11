import pickle
import torch
from Plotting import PlotError,PlotErrTime,PlotRaster,PlotWeights,ReconstructImage, PlotClusterOut,PlotClustWeight
from FashionDL import getFashionMNIST
import timeit
import os
import torch.distributions as dist


#This code trains for one epoch on a network that has already been trained.
#Would be good not to hard code these values just in case.

n_samples = 10000
log_interval = 500
image_time = 200
pw = 10

add_noise = True

# num_hidden = 10000
# num_class = 1000
# epochs_init = 4
# increase1 = 5e-2
# decay1 = 2.5e-8
# increase3 = 1e0
# decay3 = 1e-5
# beta1 = 0.9
# beta2 = 0.9
# beta3 = 0.9

# filename = 'SBWD_IT%i_NH%iNC%iEpo%iInc1_%.2eDc1_%.2eInc3_%.2eDc3_%.2eBt1_%.2eBt2_%.2eBt3_%.2e.pckl'%(image_time,
#                                                                                                 num_hidden,
#                                                                                           num_class,
#                                                                                           epochs_init,
#                                                                                           increase1,
#                                                                                           decay1,
#                                                                                           increase3,
#                                                                                           decay3,
#                                                                                           beta1,
#                                                                                           beta2,
#                                                                                           beta3)


num_hidden = 10000
num_class = 1000
epo = 7
decay3 = 5e-6
Ap = 5e-2
An = 4e-2
Taup = 3
Taun = 2e2
beta1 = 0.9
beta2 = 0.9
beta3 = 0.9




filename = 'Clamp 400 NH%iEpo%iNC%iDc3_%.2eidx60000.pckl' %(
                                                                num_hidden,
                                                               epo,
                                                               num_class,
                                                               decay3,
                                                               )

filename = 'NH10000NC1000Epo4Inc1_5.00e-02Dc1_2.50e-08Inc3_1.00e+00Dc3_1.00e-05Ap1_5.00e-02An1_2.00e-02Ap3_5.00e-02.pckl'

filestring = os.path.join(os.getcwd(),filename)
f = open(filestring,'rb')
network = pickle.load(f)
f.close()

# a = filestring.split()

epochs = 1
# num_class = network.num_class

network.w3Update = False


print("Loading MNIST test samples ...")
test_data,test_labels = getFashionMNIST(load_train=False,load_test=True,export_to_disk=False)

#Need to plot cluster weights and re-train with peri-clamp methods in both layers. but I need to know thresholds.
PlotClustWeight(network.fc3.weight.detach().numpy(),'Cluster Final weight%i.png'%(network.num_class),network.num_class)
PlotWeights(network.fc1.weight.data,
            network.fc2.weight.data.transpose(0,1),
            'W1_Final.png' ,
            'W2_Final.png',
             network.num_hidden )
print('Lif1 Thresholds')
print(network.lif1.threshold.detach()[1:100])
print(torch.max(network.lif1.threshold.detach()))
print('Lif3 Thresholds')
print(network.lif3.threshold.detach())
# sig = [0.1,0.2,0.3,0.4,0.5]#Gaussian tests
# sig = [0.025,0.05,0.1,0.2] #S and P tests. maybe just create a seperate file function
sig = [0.5]

#Think of better way for this. I think seperate files will work.
for sigma in sig:
    filename = "Gauss%.2e" %(sigma)
    directory = os.path.join(os.getcwd(),filename)
    os.mkdir(directory)
    if add_noise == True:
        #Gaussian Noise
        noise = dist.Normal(torch.zeros_like(test_data),sigma*torch.ones_like(test_data))
        test_data = test_data + noise.sample()
        test_data.clamp_(0,1)

        #Salt and pepper noise
            # prob_p = sigma#Note that probability is actually doubled.
            # prob_s = 1-prob_p
            # rdm = torch.rand((10000,784))
            # test_data[rdm<prob_p] = 1
            # test_data[rdm>prob_s] = 0
        
                      


    #Need code to add noise. I'm thinking salt and pepper noise and gaussian. and maybe combinations.


    print("Testing...")
    start_time = timeit.default_timer()
    err_rec = []
    clust_out = []
    NonRecon = 0
    #Below are just for debugging
    in_clust = []
    non_in = 0
    err_avg = 0
    Act_avg = 0
    Clu_avg = 0
    recons = []
    labels = []
    for i in range(network.num_class):
        clust_out.append([])

    for i in range(network.num_hidden):
         in_clust.append([])
    for epoch in range(epochs):
        for idx in range(n_samples):
            network.HidActivity = torch.zeros(network.num_hidden)
            network.OutActivity = torch.zeros(network.num_inputs)
            network.ClustActivity = torch.zeros(network.num_class)
            network.spkFlag = False

            error,error_scalar,spk2_recon, spk3_recon, spk1_recon= network.PresentImage(test_data[idx],image_time,pw,UpdateParams = False)
            err_rec.append(error_scalar)
            err_avg += error_scalar

            if torch.sum(network.OutActivity)==0:
                    NonRecon +=1
            elif torch.sum(network.ClustActivity)>0:
                    vals,indices = spk3_recon.min(0)
                    clust_out[indices].append(test_labels[idx])
            else:
                    vals,indices = network.mem3.max(1) #If no spike, then max membrane potential
                    clust_out[indices].append(test_labels[idx])

            #Debugging for first layer (see if it classifies well.)
            Act_avg += torch.sum(network.HidActivity)
            Clu_avg += torch.sum(network.ClustActivity)
            if torch.sum(network.HidActivity)==0:
                non_in +=1
            else:
                vals, indices = spk1_recon.min(0)
                in_clust[indices].append(test_labels[idx])

            recons.append(spk2_recon/image_time*255)
                

            if (idx+1)% log_interval == 0:
                print(
                    "Training progress: sample (%d / %d) of epoch (%d / %d) - Elapsed time: %.4f. Current Error:%d" %(
                    idx+1,
                    n_samples,
                    epoch + 1,
                    epochs,
                    timeit.default_timer()-start_time,
                    err_rec[idx]
                    )
                )
                title = 'NSY_Err_Imgs%i_Inc1%.2eDec1%.2eInc2%.2eDec2%.2eNH%iSig%.2e.svg' %(idx+1,network.increase1,network.decay1,network.increase2,network.decay2,network.num_hidden,sigma)
                # PlotError(torch.reshape(error,(28,28)),title=title)
                title = os.path.join(directory,'Lab%iNSY_TeRI_Imgs%i_Inc1%.2eDec1%.2eInc2%.2eDec2%.2eNH%iSig%.2e.png' %(test_labels[idx],
                                                                                                                        idx+1,
                                                                                                                        network.increase1,
                                                                                                                        network.decay1,
                                                                                                                        network.increase2,
                                                                                                                        network.decay2,
                                                                                                                        network.num_hidden,
                                                                                                                        sigma))
                spk2_recon = (image_time-1)*test_data[idx] - error*image_time
                ReconstructImage(spk2_recon,title, image_time)
                ReconstructImage(test_data[idx],os.path.join(directory,'Lab%iNSYInput%iSig%.2e.png' %(test_labels[idx],idx+1,sigma)),1)
                # print('Hidden Layer Activity')
                # print(torch.sum(network.HidActivity))
                # print('Recontruction Layer Activity')
                # print(torch.sum(network.OutActivity))
                # print('Cluster Layer Activity')
                # print(torch.sum(network.ClustActivity))
                # print('Hidden Activity')
                # print(network.HidActivity)
                # print('Hidden Threshold')
                # print(network.lif1.threshold.detach())
                # print('Output Threshold')
                # print(network.lif2.threshold.detach())
    RT = torch.stack(recons)
    torch.save((RT,test_labels),os.path.join(directory,'test.pt'))
    PlotErrTime(err_rec,'NSY Test Error Time.png')
    PlotClusterOut(clust_data=clust_out,title = 'NSY_Clustering Plot_Inc3%.2eDec3%.2eAp3%.2eAn3%.2eTauP3%.2eTauN3%.2eNC%iSig%.2e.svg' %(
        network.increase3,
        network.decay3,
        network.params3[0],
        network.params3[1],
        network.params3[2],
        network.params3[3],
        network.num_class,
        sigma),num_class=network.num_class)
    print('Debugging')
    PlotClusterOut(clust_data = in_clust, title = 'Debugging.png',num_class = network.num_hidden)
    print('Sigma')
    print(sigma)
    print('NonRecon')
    print(NonRecon)
    print('NumClassif')
    print(network.num_class)
    print("None in")
    print(non_in)
    print('Average MSE')
    print(err_avg/10000)
    print('Average Hidden Activity')
    print(Act_avg/10000)
    print('Cluster AVerage Accuracy')
    print(Clu_avg/10000)



    


