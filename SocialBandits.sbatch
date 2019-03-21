#!/bin/bash
#set a job name 
#SBATCH --job-name=SocialBandits
#a file for job output, you can check job progress
#SBATCH --output=./IOERR/SocialBandits_%j.out
# a file for errors from the job
#SBATCH --error=./IOERR/SocialBandits_%j.err
#time you think you need: default is one day 
#in minutes in this case, hh:mm:ss
#SBATCH --time=48:00:00
#number of cores you are requesting 
#SBATCH --cpus-per-task=1
#memory you are requesting
#SBATCH --mem=10Gb


module load anaconda3/3.6
module load python/3.6.6

srun python3 SocialBandits.py $@
