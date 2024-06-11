import itertools
import os
import subprocess
import uuid
import time

Ap1 = [0.05]#0.05
Ap2 = [0.05]
An1 = [0.03]#0.03#0.004 was 0.04 before w1 reduction
An2 = [0.005]
beta1 = [0.9]
beta2 = [0.9] #Lowering beta2 seems to work.
latency = [10]
decay1 = [5e-8]#1e-5 is tested thoroughly
decay2 = [5e-5]
TauP2 = [30]
increase1 = [5e-2,5e-3,5e-4,5e-5]
increase2 = [1e-2]
TauP1 = [20]
TauN1 = [200]
pw = [10]
num_hidden = [5000]
train_epochs = [7]
increase3 = [5e-2]
decay3 = [2.5e-6]
Ap3 = [0.2]
An3 = [0.12]
TauP3 = [20]
TauN3 = [200]
num_class = [100]#I think above only works for 100.
lif1_scale = [0.999]
lif3_scale = [0.9]
combinations = list(
    itertools.product(Ap1, Ap2, An1, An2, beta1, beta2, latency,decay1,decay2,TauP2,increase1,increase2,TauP1,TauN1,pw,num_hidden,train_epochs,
                      increase3,decay3,Ap3,An3,TauP3,TauN3,num_class,lif1_scale,lif3_scale)
)

cwd = os.getcwd()
for combination in combinations:
    d = {
        "Ap1": combination[0],
        "Ap2": combination[1],
        "An1": combination[2],
        "An2": combination[3],
        "beta1":combination[4],
        "beta2":combination[5],
        "latency":combination[6],
        "decay1":combination[7],
        "decay2":combination[8],
        "TauP2":combination[9],
        "increase1":combination[10],
        "increase2":combination[11],
        "TauP1":combination[12],
        "TauN1":combination[13],
        "pw":combination[14],
        "num_hidden":combination[15],
        "train_epochs":combination[16],
        "increase3":combination[17],
        "decay3":combination[18],
        "Ap3":combination[19],
        "An3":combination[20],
        "TauP3":combination[21],
        "TauN3":combination[22],
        "num_class":combination[23],
        "lif1_scale":combination[24],
        "lif3_scale":combination[25]

    }
    args = ""
    for key in d:
        args = args + ' --' + key + ' ' + str(d[key])

    bash_script = """#!/bin/bash -l
#Edit this script to suit your purposes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=500G
#SBATCH --job-name=Bunya_AutoEncode
#SBATCH --time=70:00:00
#SBATCH --partition=general
#SBATCH --account=a_rahimi
#SBATCH -o "%s"
#SBATCH -e "%s"


module load anaconda3/2022.05
source /sw/auto/rocky8.6/epyc3/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate myenv
cd /scratch/user/benwalters/STDP-Autoencoder-Network
python Grid.py%s

""" %( os.path.join(cwd, 'outInc1_%.2e_test.txt' %(d["increase1"])), os.path.join(cwd, 'error.txt'),args)#This is to edit the above script.


    myuuid = str(uuid.uuid4())
    with open(os.path.join(os.getcwd(), "%s.sh" % myuuid), "w+") as f:
        f.writelines(bash_script)

    res = subprocess.run("sbatch %s.sh" % myuuid, capture_output=True, shell=True)
    print(args)
    print(res.stdout.decode())
    os.remove(os.path.join(os.getcwd(), "%s.sh" % myuuid))
    time.sleep(2)
