#!/bin/bash
#set a job name 
#SBATCH --job-name=SocialBandits
#a file for job output, you can check job progress
#SBATCH --output=SocialBandits_%j.out
# a file for errors from the job
#SBATCH --error=SocialBandits_%j.err
#time you think you need: default is one day 
#in minutes in this case, hh:mm:ss
#SBATCH --time=24:00:00
#number of cores you are requesting 
#SBATCH --cpus-per-task=1
#memory you are requesting
#SBATCH --mem=10Gb

module load python/3.7.1


srun python3 SocialBandits.py $@
