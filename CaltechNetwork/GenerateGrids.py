import itertools
import os
import subprocess
import uuid
import time
import sys
import subprocess
import torch




Ap1 = [0.06]
Ap2 = [0.1]
An1 = [0.03]#0.02
An2 = [0.001]
inc1 = [1e1]#1e-1
dec1 = [5e-3]#5e-5 was previous working result.
inc2 = [0]#0
dec2 = [0]#0
TauP1 = [22]#33
image_time = [200]#500
num_hidden = [1500]#800

combinations = list(
    itertools.product(Ap1, Ap2, An1, An2,inc1,dec1,inc2,dec2,TauP1,image_time,num_hidden)
)

cwd = os.getcwd()
for combination in combinations:
    d = {
        "Ap1": combination[0],
        "Ap2": combination[1],
        "An1": combination[2],
        "An2": combination[3],
        "increase1":combination[4],
        "decay1":combination[5],
        "increase2":combination[6],
        "decay2":combination[7],
        "TauP1":combination[8],
        "image_time":combination[9],
        "num_hidden":combination[10]
    }
    args = ""
    for key in d:
        args = args + ' --' + key + ' ' + str(d[key])

    bash_script = """#!/bin/bash -l
#Edit this script to suit your purposes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=100G
#SBATCH --job-name=Bunya_AutoEncode
#SBATCH --time=30:00:00
#SBATCH --partition=general
#SBATCH --account=a_rahimi
#SBATCH -o "%s"
#SBATCH -e "%s"


module load anaconda3/2022.05
source /sw/auto/rocky8.6/epyc3/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate myenv
cd /scratch/user/benwalters/CaltechNetwork
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/user/benwalters/conda_env/lib
python Grid.py%s


""" % ( os.path.join(cwd, 'Calout.txt'), os.path.join(cwd, 'Calerror.txt'),args)#This is to edit the above script.


    myuuid = str(uuid.uuid4())
    with open(os.path.join(os.getcwd(), "%s.sh" % myuuid), "w+") as f:
        f.writelines(bash_script)

    res = subprocess.run("sbatch %s.sh" % myuuid, capture_output=True, shell=True)
    print(args)
    print(res.stdout.decode())
    os.remove(os.path.join(os.getcwd(), "%s.sh" % myuuid))
    time.sleep(2)
