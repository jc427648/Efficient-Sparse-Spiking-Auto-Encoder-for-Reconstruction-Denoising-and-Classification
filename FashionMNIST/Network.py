import torch
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc

from STDP import GetSTDP
from Plotting import PlotError, PlotWeights






class Net(nn.Module):
    def __init__(self,beta1,beta2,num_hidden,num_inputs,decay1,decay2,increase1,increase2,params1,params2,latency,clustering,params3,increase3,decay3,beta3,num_class):
        super().__init__()

        # Initialize layers
        # self.W1 = torch.rand((num_hidden,num_inputs))
        self.lif1 = snn.Leaky(beta = beta1,learn_threshold = True,threshold = torch.ones(num_hidden),inhibition=True,reset_mechanism="zero")
        self.lif2 = snn.Leaky(beta = beta2,learn_threshold=True,threshold = torch.ones(num_inputs), inhibition = False) #not sure if should be true or false.
        # self.W2 = torch.rand((num_inputs,num_hidden))

        self.fc1 = nn.Linear(num_inputs,num_hidden, bias = False)
        self.fc2 = nn.Linear(num_hidden,num_inputs,bias = False)
        init_wt1 = torch.rand(num_hidden,num_inputs)
        init_wt2 = torch.rand(num_inputs,num_hidden) 
        with torch.no_grad():
                self.fc1.weight.copy_(init_wt1)
                self.fc2.weight.copy_(init_wt2)

        self.params1 =  params1 #parameters for STDP window btwn input and hidden
        self.params2 = params2
        self.STDP1 = GetSTDP(parameters = self.params1,mode = 'exponential')
        self.STDP2 = GetSTDP(parameters = self.params2, mode = 'exponential')
        self.latency = latency
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden

        self.rho = torch.ones(num_inputs)
        self.b = 0.15
        self.decay1 = decay1
        self.decay2 = decay2
        self.increase1 = increase1
        self.increase2 = increase2

        self.Lif1Act = torch.zeros(num_hidden)
        self.Lif2Act = torch.zeros(num_inputs)
        InitThresh1 = 100*torch.ones(num_hidden)
        InitThresh2 = 5.0*torch.ones(num_inputs)

        self.clust  = False
        self.w3Update = True

        self.lif3Max = 200
        self.lif3Min = 50

        self.lif1Max = 400
        self.lif1Min = 150

        with torch.no_grad():
            self.lif1.threshold.copy_(InitThresh1)
            self.lif2.threshold.copy_(InitThresh2)

        if clustering == True:
            self.num_class = num_class
            self.params3 = params3
            self.clust = clustering
            self.STDP3 =  GetSTDP(mode = 'exponential', parameters = self.params3)
            self.lif3 = snn.Leaky(beta = beta3,learn_threshold = True,threshold = 100*torch.ones(num_class),inhibition=True)
            self.fc3 = nn.Linear(num_inputs,num_class,bias = False)
            init_wt3 = torch.rand(num_class,num_inputs)
            with torch.no_grad():
                 self.fc3.weight.copy_(init_wt3)
            self.mem3 = torch.zeros(num_class)
            self.increase3 = increase3
            self.decay3 = decay3
            self.ClustActivity = torch.zeros(num_class)
            self.spkFlag = False
            self.spk3 = torch.zeros(num_class)
        
 

    def step(self, x, mem1, mem2):

        # # Initialize hidden states at t=0
        # mem1 = self.lif1.init_leaky()
        # mem2 = self.lif2.init_leaky()
        #NEED TO CONSIDER IF THIS IS INITIALISED OR RECORDED. I DON'T KNOW.
        # Record the final layer and hidden layer

        with torch.no_grad():
            cur1 = self.fc1(x) 
            spk1, mem1 = self.lif1(cur1, mem1)
            

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            self.HidActivity = self.HidActivity + spk1
            self.OutActivity = self.OutActivity + spk2

            if self.clust == True:
                    if self.spkFlag ==False | (self.spkFlag == True):
                        cur3 = self.fc3(spk2)
                        self.spk3,self.mem3 = self.lif3(cur3,self.mem3)
                        self.ClustActivity += self.spk3.squeeze()
                        if torch.sum(self.spk3)>0:
                            self.spkFlag = True
                            self.mem3.zero_()

        
        return spk1, mem1, spk2, mem2, self.spk3
    
    def CalcError(self,spk2_recon,image,time):
        #Image is between 0 and 1.
        spk2_recon = spk2_recon - self.latency #account for network latency
        #These are the actual input spike times.

        error = (image-spk2_recon)/time
        error1 = torch.square(error)
        error_scalar = torch.sum(error1)

        return error, error_scalar
    # @profile   
    def PresentImage(self,image,time,pw,UpdateParams=True):
        #Maybe introduce a short timespan in which the neurons relax between images, that way no learning occurs.
        #Or just do a straight hard reset and see what happens.MIGHT NEED A VARIABLE FOR TRAIN AND TEST.
        
        #Need to unpack the image into a 1x784
        image = (time)*torch.flatten(image) #This now gives idx when each pixel should spike. This should be entered as input
        #Then need to create pulse height timestepsx784.
        input_spikes = torch.zeros(time,self.num_inputs)
        for i in range(image.shape.numel()):
            input_spikes[image[i].int():image[i].int()+pw,i] = 1
        # input_spikes[-1,:] = 0

        
        # inputspikes[]
        #Reset membrane potential
        mem1 =torch.zeros(self.num_hidden)#self.lif1.init_leaky()
        mem2 = torch.zeros(self.num_inputs)
        self.mem3.zero_()
        
        spk1_history = torch.zeros(self.num_hidden)#.to_sparse()
        spk2_history = torch.zeros(self.num_inputs)
        spk3_history = torch.zeros(self.num_class)

        spk1_recon = torch.zeros(self.num_hidden)
        spk2_recon = torch.zeros(self.num_inputs)# used to calculate the earliest spike times for reconstruction
        spk3_recon = torch.zeros(self.num_class)
        spk1 = torch.zeros(self.num_hidden)
        spk2 = torch.zeros(self.num_inputs)
        spk3 = torch.zeros(self.num_class)
        
        
        for t  in range(time):
            spk1,mem1,spk2,mem2,spk3 = self.step(input_spikes.unsqueeze(1)[t], mem1=mem1,mem2=mem2)
            #update neuron activity and history
            #Now calculate changes in post-synaptic spike times and changes in synaptic weights.  
            #Use the t-variable to calculate the time. can also use image. spk1 is spikes of hidden layer. We need to record these in order to calculate delta t.
            #spk1*t gives either the most recent version. We could then compare to see which one is greater.
            spk1_history.copy_(spk1_history.maximum(spk1*(t+1)).squeeze()) #We could use this variable for homeostatic plasticity as well.
            spk2_history.copy_(spk2_history.maximum(spk2*(t+1)).squeeze()) #elementwise maximum
            spk3_history.copy_(spk3_history.maximum(spk3*(t+1)).squeeze())

            spk1_recon.copy_(spk1_recon.where(spk1_recon!=0,spk1_history))
            spk2_recon.copy_(spk2_recon.where(spk2_recon!=0,spk2_history))
            spk3_recon.copy_(spk3_recon.where(spk3_recon!=0,spk3_history))

            

            if UpdateParams == True:
                 #decay parameter too small, can't actually decrement the number.
                NewThresh1 = self.lif1.threshold.data - self.decay1 + self.increase1*spk1
                # NewThresh2 = self.lif2.threshold.data - self.decay2 +self.increase2*spk2
                NewThresh3 = self.lif3.threshold.data - self.decay3 + self.increase3*spk3
                NewThresh3.clamp_(self.lif3Min,self.lif3Max)
                NewThresh1.clamp_(self.lif1Min,self.lif1Max)
                with torch.no_grad():
                    self.lif1.threshold.copy_(NewThresh1.squeeze())
                    # self.lif2.threshold.copy_(NewThresh2.squeeze())
                    self.lif3.threshold.copy_(NewThresh3.squeeze())
                     
            
            

        spk2_recon[spk2_recon==0] = time
        spk1_recon[spk1_recon==0] = time
        spk3_recon[spk3_recon==0] = time
        if UpdateParams == True:

            delta_t1 = (spk1_recon).unsqueeze(1).repeat(1,self.num_inputs) - image.unsqueeze(0).repeat(self.num_hidden,1) 
            delta_t1 = delta_t1*((self.HidActivity.transpose(0,1)>0).repeat(1,self.num_inputs))
            delta_t2 = spk2_recon.unsqueeze(1).repeat(1,self.num_hidden) - spk1_recon.unsqueeze(0).repeat(self.num_inputs,1)
            


            
            delta_w1 = torch.from_numpy(self.STDP1[1,(delta_t1+self.params1[5]-1).int()])
            delta_w2 = torch.from_numpy(self.STDP2[1,(delta_t2+self.params2[5]-1).int()])
            #Account for relevance
            Zeta = 1*((time - spk1_recon)*(self.HidActivity>0)/time) #Term to decide which neuron connections are more important than others
            self.rho = 1*(spk2_recon - self.latency - image)/time +self.b
            

            #Add some weight dependancy
            # WD = self.fc1.weight.detach()*(1-self.fc1.weight.detach())
            # delta_w1.mul_(WD)           
            NewWeights1 = (delta_w1 + self.fc1.weight.detach()).detach()#.clamp(0,1) #We're simulating weights between 0 and 1
            
            #comment out clamp for softbound
            #Multiply delta_w2 by error and pre-synaptic modulation
            delta_w2.mul_(self.rho.unsqueeze(1).repeat(1,self.num_hidden)*Zeta.repeat(self.num_inputs,1))

            NewWeights2 = torch.add(delta_w2,self.fc2.weight.detach())
            NewWeights2.clamp_(0,1)
            
            
            #Update neuronal thresholds
            NewThresh3 = self.lif3.threshold.data - self.decay3*time
            NewThresh3.clamp_(self.lif3Min,self.lif3Max)
            NewThresh1.clamp_(self.lif1Min,self.lif1Max)
            
            with torch.no_grad():
                self.fc1.weight.copy_(NewWeights1)
                self.fc2.weight.copy_(NewWeights2)
                self.lif3.threshold.copy_(NewThresh3.squeeze())

        
        error, error_scalar = self.CalcError(spk2_recon=spk2_recon,image=image,time=time)
        
        if self.w3Update==True:
            delta_t3 = (spk3_recon.repeat(self.num_inputs,1).transpose(0,1) - spk2_recon.repeat(self.num_class,1))
            delta_w3 = torch.from_numpy(self.STDP3[1,(delta_t3+self.params3[5]-1).int()])

            #Add weight dependancy
            # WD = self.fc3.weight.detach()*(1-self.fc3.weight.detach())
            
            NewWeights3 = torch.add(delta_w3,self.fc3.weight.detach())
            # NewWeights3.clamp_(0,1)#Commented out for Soft-bound instead of hard-bound

            with torch.no_grad():
                self.fc3.weight.copy_(NewWeights3)
        
        return error,error_scalar, spk2_recon, spk3_recon, spk1_recon
        








