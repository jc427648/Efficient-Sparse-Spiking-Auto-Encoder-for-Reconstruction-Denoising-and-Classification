import torch.nn as nn
import torch



#Here is a list of functions that I think we'l require:


#Functions for loading and de-loading MNIST (take from other project)



#Functions for initialising STDP window. Use exponential or different shapes etc. Have parameters input into the network.


#We need functions for updating the synaptic weight. This can be similar to previous I think.

#We need functions for the neurons as well. I was thinking of using torch and snntorch to help with this.

#We need to use anaconda as well I think in order to use the HPC.

#Need a function for calculating the error for the auto encoder. I think we present the image, propagate through the network and then at the end calculate the error based o
#on all of the timings or rates that we record. The frontiers paper with ihlemenni goes through how to calculate an error with a latency, and we could introduce this latency
#as a hyperparameter. This works for more of a TTFS scheme, which would probably be more efficient than a poissonian scheme. If we encode 
#the image as poissonian, then the output measurement also needs to be poissonian. If our input is TTFS, then our output should also be TTFS
#To be comparable. We can investigate both, but remember our initial code binarises the image. Maybe start with TTFS, and use rule proposed.


#I definitely think that we should just use three layers for the moment. most other works seem to do this.

#Let's run some test code to understand how it works. Can use nn.sequential to define sequential layers.

#May need homeostatic mechanisms as well. Don't forget threshold as hyperparameter.




# m = nn.Linear(2,1) #Use nn.functional.linear to be able to control the weights. I think weights might need to be a seperate variable.
# input = torch.randn(3,2)
# output = m(input)
# a= 1
