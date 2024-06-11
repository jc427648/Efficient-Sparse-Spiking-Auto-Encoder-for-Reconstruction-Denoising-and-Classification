import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt


from STDP import GetSTDP
from Plotting import PlotError, PlotWeights




class Net(nn.Module):
    def __init__(self,beta1,beta2,num_hidden,num_inputs,decay1,decay2,increase1,increase2,params1,params2,latency):
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
        self.decay1 = decay1
        self.decay2 = decay2
        self.increase1 = increase1
        self.increase2 = increase2

        self.Lif1Act = torch.zeros(num_hidden)
        self.Lif2Act = torch.zeros(num_inputs)
        InitThresh1 = 20*torch.ones(num_hidden)
        InitThresh2 = 2*torch.ones(num_inputs)#2 for decent results.
        with torch.no_grad():
            self.lif1.threshold.copy_(InitThresh1)
            self.lif2.threshold.copy_(InitThresh2)
        
 

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
            
        



        return spk1, mem1, spk2, mem2
    
    def CalcError(self,spk2_recon,image,time):
        #Image is between 0 and 1.
        spk2_recon = spk2_recon - self.latency #account for network latency
        #These are the actual input spike times.

        error = (image-spk2_recon)/time #200 is image time
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

        #Cut Dead weight
        # inputspikes[]
        #Reset membrane potential
        mem1 =torch.zeros(self.num_hidden)#self.lif1.init_leaky()
        mem2 = torch.zeros(self.num_inputs)
        
        spk1_history = torch.zeros(self.num_hidden)#.to_sparse()
        spk2_history = torch.zeros(self.num_inputs)

        spk1_recon = torch.zeros(self.num_hidden)
        spk2_recon = torch.zeros(self.num_inputs)# used to calculate the earliest spike times for reconstruction

        spk1 = torch.zeros(self.num_hidden)
        spk2 = torch.zeros(self.num_inputs)
        
        
        for t  in range(time):
            spk1,mem1,spk2,mem2 = self.step(input_spikes.unsqueeze(1)[t], mem1=mem1,mem2=mem2)
            #update neuron activity and history
            #Now calculate changes in post-synaptic spike times and changes in synaptic weights.  
            #Use the t-variable to calculate the time. can also use image. spk1 is spikes of hidden layer. We need to record these in order to calculate delta t.
            #spk1*t gives either the most recent version. We could then compare to see which one is greater.
            spk1_history.copy_(spk1_history.maximum(spk1*t).squeeze()) #We could use this variable for homeostatic plasticity as well.
            spk2_history.copy_(spk2_history.maximum(spk2*t).squeeze()) #elementwise maximum

            spk1_recon.copy_(spk1_recon.where(spk1_recon!=0,spk1_history))
            spk2_recon.copy_(spk2_recon.where(spk2_recon!=0,spk2_history))
            
            #I also need the earlies spike times for spk2 to reconstruct the image. or I make lif2 spike once only per image.

            if UpdateParams == True:

                #Update neuronal thresholds
                NewThresh1 = self.lif1.threshold.data - self.decay1 + self.increase1*spk1
                NewThresh2 = self.lif2.threshold.data - self.decay2 +self.increase2*spk2
                
                

                # NewThresh1 = torch.clamp(NewThresh1,0,100)
                # NewThresh2 = torch.clamp(NewThresh2,0.5,2) #Necessary, so that thresholds don't fight weights.
                
                with torch.no_grad():
                    # self.fc1.weight.copy_(NewWeights1)
                    # self.fc2.weight.copy_(NewWeights2)
                    self.lif1.threshold.copy_(NewThresh1.squeeze())
                    self.lif2.threshold.copy_(NewThresh2.squeeze())
                # PlotWeights(self.fc1.weight.data, self.fc2.weight.data.transpose(0,1),'W1 %i.png'%(t),'W2 %i.png'%(t))
                



        #adjust for non-spiking neurons
        spk2_recon[spk2_recon==0] = time
        spk1_recon[spk1_recon==0] = time
        if UpdateParams == True:

            #You will want the if statement to speed up the process, for now ignoring.
            delta_t1 = (spk1_recon).unsqueeze(1).repeat(1,self.num_inputs) - image.unsqueeze(0).repeat(self.num_hidden,1) #delta t values for image to hidden layer
            delta_t1 = delta_t1*((self.HidActivity.transpose(0,1)>0).repeat(1,self.num_inputs))
            delta_t2 = spk2_recon.unsqueeze(1).repeat(1,self.num_hidden) - spk1_recon.unsqueeze(0).repeat(self.num_inputs,1)# delta t values for hidden layer to output layer.
            


            #Can simply convert delta_t to idx via addition, and then substitue idx for delta W.
            delta_w1 = torch.from_numpy(self.STDP1[1,(delta_t1+self.params1[5]-1).int()])
            delta_w2 = torch.from_numpy(self.STDP2[1,(delta_t2+self.params2[5]-1).int()])
            #Account for relevance
            Relevance = 1*((time - spk1_recon)*(self.HidActivity>0)/time) #Term to decide which neuron connections are more important than others
            self.rho = 1*(spk2_recon - self.latency - image)/time +0.15
            # delta_w2 = delta_w2 + Relevance

            #Update the weights
            # DelW1 = torch.multiply(delta_w1.unsqueeze(1),spk1) #Creates a vector the same size as the weights vector.
            NewWeights1 = (delta_w1 + self.fc1.weight.detach()).detach().clamp(0,1) #We're simulating weights between 0 and 1
            
            
            
            delta_w2.mul_(self.rho.unsqueeze(1).repeat(1,self.num_hidden)*Relevance.repeat(self.num_inputs,1))

            NewWeights2 = torch.add(delta_w2,self.fc2.weight.detach())
            NewWeights2.clamp_(0,1)

            
            with torch.no_grad():
                self.fc1.weight.copy_(NewWeights1)
                self.fc2.weight.copy_(NewWeights2)
                
        error, error_scalar = self.CalcError(spk2_recon=spk2_recon,image=image,time=time)
        # print(self.rho)
        
        #Now calculate the update for the weight modulation factor
        # error = -error.reshape(self.num_inputs)/time #time is the max that error can be. Since STDP window is arbitrary, it doesn't matter how this is set.
        return error,error_scalar
        







# #Further tests:
# beta1 = 0.1
# beta2 = 0.9

# num_hidden = 100
# num_inputs = 784 #FOR MNIST images

# #Homeostatic mechanisms:
# decay1 = 0.05
# increase1 = 5

# decay2 = 0.05
# increase2 = 5

# params1 = [0.005,0.001,20,200,1000,1000]#Minimum values is 0.005 and 0.001. May want to go higher than this. I think time is ok.
# params2 = [0.5,0.1,50,200,1000,1000] #Make last two parameters the time length of the image.

# #latency for error calculations:
# latency = 31 #timesteps.
# #all above variables should go into the init def.
# net = Net(beta1=beta1,
#           beta2=beta2,
#           num_hidden=num_hidden,
#           num_inputs=num_inputs,
#           decay1=decay1,
#           decay2=decay2,
#           increase1=increase1,
#           increase2=increase2,
#           params1=params1,
#           params2=params2,
#           latency=latency)
# # PlotWeights(net.fc1.weight.data,net.fc2.weight.data,'initweights1.png','initweights2.png')
# images = torch.load('train_images.pt')
# err_rec = []

# #Should calculate simulation time.
# for idx in range(100):
#     error, error_scalar = net.PresentImage(images[idx],1000,10)
#     # error,error_scalar = net.CalcError(spk2_recon=spk2_recon,image = images[0],time = 1000, pw = 30)
#     err_rec.append(error_scalar.item())
#     if idx == 20:
#         PlotError(error=error.reshape(28,28),title = '20 Error.png')
#         PlotWeights(net.fc1.weight.data,net.fc2.weight.data.transpose(0,1),'20 W1.png','20 W2.png')
#         print(20)
#     elif idx == 50:
#         PlotError(error=error.reshape(28,28),title = '50 Error.png')
#         PlotWeights(net.fc1.weight.data,net.fc2.weight.data.transpose(0,1),'50 W1.png','50 W2.png')
#         print(50)
#     elif idx == 80:
#         PlotError(error=error.reshape(28,28),title = '80 Error.png')
#         PlotWeights(net.fc1.weight.data,net.fc2.weight.data.transpose(0,1),'80 W1.png','80 W2.png')
#         print(80)
    
# PlotError(error=error.reshape(28,28), title = 'Final Error')
# PlotWeights(net.fc1.weight.data,net.fc2.weight.data.transpose(0,1),'Final W1.png','Final W2.png')
# print(err_rec)
# print(net.lif1.threshold.data)
# print(net.lif2.threshold.data)
# print(net.rho)
# plt.figure()
# plt.plot(net.STDP1[0,:],net.STDP1[1,:])
# plt.savefig('STDP window.png')

# # inp_spks = 
# a = 1
# x = 0.5*torch.ones(num_inputs)
# spk1,mem1,spk2,mem2 = net.step(x,net.lif1.init_leaky(),net.lif2.init_leaky()) #The init_leaky can be used to initialise the membrane potentials.





# with torch.no_grad():
#     net.fc1.weight.copy_(torch.ones((100,784)))
# a=1 #Above code allows me to change the synaptic weights. Now we are going somewhere.


#Having just mucked around with this, I've found that everything is normalise, including the membrane potential and threshold.


#I think the rest of the code has to do this: convert the image into a current for a particular time, Apply said current, and do for all timesteps. Then the rest is calculate error.
#The only problem I see is how to do






