#These functions will generate and calculate the STDP updates.

import numpy as np
import torch


def GetSTDP(mode,parameters):
    
    if mode == 'exponential':
        Ap = parameters[0]
        An = parameters[1]
        tauP = parameters[2]
        tauN = parameters[3]
        PosLength = parameters[4]
        NegLength = parameters[5]

        DelT = np.linspace(-NegLength,PosLength,NegLength+PosLength+1)
        WUpdate = np.zeros(PosLength+NegLength+1)
        #...
        WUpdate[0:(NegLength-1)] = -An*np.exp(DelT[0:(NegLength-1)]/tauN)
        WUpdate[(NegLength+1):(PosLength+ NegLength+1)] = Ap*np.exp(-DelT[(NegLength+1):(PosLength+NegLength+1)]/tauP)

    elif mode == 'Layer2':
        Ap = parameters[0]
        An = parameters[1]
        tauP = parameters[2]
        tauN = parameters[3]
        PosLength = parameters[4]
        NegLength = parameters[5]

        DelT = np.linspace(-NegLength,PosLength,NegLength+PosLength+1)
        WUpdate = np.zeros(PosLength+NegLength+1)
        WUpdate[(NegLength+1):(PosLength + NegLength+1)] = Ap*np.exp(DelT[(NegLength+1):(PosLength+NegLength+1)]/tauP)
        WUpdate[0:(NegLength-1)] = -An*np.exp(DelT[0:(NegLength-1)]/tauN)



    Window = np.zeros((2,PosLength + NegLength+1))
    Window[0,:] = DelT
    Window[1,:] = WUpdate

    return Window

# def CalcSTDP(PreTimes,PostTimes,weight):


