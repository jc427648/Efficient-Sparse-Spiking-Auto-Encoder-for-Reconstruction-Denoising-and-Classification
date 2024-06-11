#!/bin/bash -l
#Edit this script to suit your purposes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --job-name=Bunya_AutoEncode
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --account=a_rahimi
#SBATCH -o slurm.out
#SBATCH -e slurm.error
#SBATCH --constraint=epyc3
#SBATCH --batch=epyc3


module load anaconda3/2022.05
source /sw/auto/rocky8.6/epyc3/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate myenv
cd /scratch/user/benwalters/STDP-Autoencoder-Network

python GenerateGrids.py