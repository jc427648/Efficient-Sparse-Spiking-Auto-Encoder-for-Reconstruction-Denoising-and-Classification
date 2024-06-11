from Network import Net 
from Main import Train,Test
from Plotting import PlotWeights,PlotError

import itertools
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import numpy as np
import torch


def run(
        beta1,
        beta2,
        num_hidden,
        num_inputs,
        decay1,
        decay2,
        increase1,
        increase2,
        Ap1,
        Ap2,
        An1,
        An2,
        TauP1,
        TauP2,
        TauN1,
        TauN2,
        latency,
        image_time,
        pw,
        train_epochs,
        clustering,
        Ap3,
        An3,
        TauP3,
        TauN3,
        increase3,
        decay3,
        beta3,
        num_class,
        lif1_scale,
        lif3_scale
):
    params1 = [Ap1,An1,TauP1,TauN1,image_time,image_time]
    params2 = [Ap2,An2,TauP2,TauN2,image_time,image_time]
    params3 = [Ap3,An3,TauP3,TauN3,image_time,image_time]
    network = Net(beta1=beta1,
                beta2=beta2,
                num_hidden=num_hidden,
                num_inputs=num_inputs,
                decay1=decay1,
                decay2=decay2,
                increase1=increase1,
                increase2=increase2,
                params1=params1,
                params2=params2,
                latency=latency,
                clustering = clustering,
                params3 = params3,
                increase3 = increase3,
                decay3 = decay3,
                beta3 = beta3,
                num_class=num_class)
    plt.figure()
    plt.plot(network.STDP2[0,:],network.STDP2[1,:])
    plt.savefig('STDP window.png')
    network = Train(network=network,
                    epochs=train_epochs,
                    image_time=image_time,
                    pw=pw,
                    lif1_scale=lif1_scale,
                    lif3_scale=lif3_scale)
    string1 = 'Sc1_%.2eW1_Ap%.2eAn%.2eTp%.2eTn%.2ebt1%.2edc1%.2einc1%.2elat%.2ePW%.2eIT%.2e.svg' %(
        lif1_scale,
        Ap1,
        An1,
        TauP1,
        TauN1,
        beta1,
        decay1,
        increase1,
        latency,
        pw,
        image_time
    )
    string2 = 'W2_Ap%.2eAn%.2eTp%.2eTn%.2ebt1%.2edc1%.2einc1%.2elat%.2ePW%.2eIT%.2e.svg' %(
        Ap2,
        An2,
        TauP2,
        TauN2,
        beta2,
        decay2,
        increase2,
        latency,
        pw,
        image_time
    ) 
    PlotWeights(network.fc1.weight.data, network.fc2.weight.data.transpose(0,1),string1,string2, network.num_hidden)
    err_rec = Test(network=network,image_time=image_time,pw=pw)
    err_avg = np.average(torch.stack(err_rec).detach().numpy())
    df = pd.read_csv(os.path.join(os.getcwd(), "grid_out.csv"))
    df = pd.concat([df,pd.DataFrame({"beta1":beta1,
         "beta2":beta2,
         "num_hidden":num_hidden,
         "num_inputs":num_inputs,
         "decay1":decay1,
         "decay2":decay2,
         "increase1":increase1,
         "increase2":increase2,
         "params1":params1,
         "params2":params2,
         "latency":latency,
         "Pulse Width":pw,
         "Image time":image_time,
         "Train epochs":train_epochs,
         "err_avg":err_avg        
        })],ignore_index = True)
    df.to_csv(os.path.join(os.getcwd(),"grid_out.csv"),index = False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta1",type=float,default = 0.9)
    parser.add_argument("--beta2",type = float, default = 0.9)
    parser.add_argument("--num_hidden",type = int,default=500)
    parser.add_argument("--num_inputs",type = int, default=784)
    parser.add_argument("--decay1",type = float,default = 1e-4) 
    parser.add_argument("--decay2",type = float,default = 1e-4)
    parser.add_argument("--increase1",type=float,default = 1e0)
    parser.add_argument("--increase2",type = float, default = 1e-2)
    parser.add_argument("--Ap1",type = float, default = 0.12)
    parser.add_argument("--Ap2",type = float,default = 0.3)#0.0005
    parser.add_argument("--An1",type = float,default = 0.06)
    parser.add_argument("--An2",type = float,default = 0.002)
    parser.add_argument("--TauP1", type = float,default = 20)
    parser.add_argument("--TauP2",type = float,default = 30)
    parser.add_argument("--TauN1",type = float, default = 200)
    parser.add_argument("--TauN2",type = float,default = 200)
    parser.add_argument("--latency",type = int, default = 10)
    parser.add_argument("--pw",type = int, default = 10)
    parser.add_argument("--image_time",type = int,default = 200)
    parser.add_argument("--train_epochs",type = int, default = 2)
    parser.add_argument("--clustering",type = bool,default = True)
    parser.add_argument("--Ap3",type = float,default = 0.012)
    parser.add_argument("--An3", type = float, default = 0.006)
    parser.add_argument("--TauP3",type = float,default = 20)
    parser.add_argument("--TauN3",type = float, default = 200)
    parser.add_argument("--beta3",type = float, default = 0.9)
    parser.add_argument("--increase3", type = float, default = 1e-1)
    parser.add_argument("--decay3", type = float, default = 5e-5)
    parser.add_argument("--num_class",type = int,default = 10)
    parser.add_argument("--lif1_scale",type = float, default = 0.999)
    parser.add_argument("--lif3_scale",type = float, default = 0.9)

    args = parser.parse_args()
    if os.path.exists(os.path.join(os.getcwd(),"grid_out.csv")):
        df = pd.read_csv(os.path.join(os.getcwd(),"grid_out.csv"))
    else:
        df = pd.DataFrame(
            columns = [
            "beta1",
            "beta2",
            "num_hidden",
            "decay1",
            "decay2",
            "increase1",
            "increase2",
            "Ap1",
            "Ap2",
            "An1",
            "An2",
            "TauP1",
            "TauP2",
            "TauN1",
            "TauN2",
            "latency",
            "pw",
            "image_time",
            "train_epochs"
            ]
        )
        df.to_csv(os.path.join(os.getcwd(),"grid_out.csv"),index= False)
    
    run(**vars(args))