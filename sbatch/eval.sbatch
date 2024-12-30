#!/bin/bash

#SBATCH --partition=hpe
#SBATCH --nodelist=hpe162
#SBATCH --nodes=1                      
#SBATCH --job-name=X-ray_Detector
#SBATCH --ntasks=1                    
#SBATCH --ntasks-per-node=1        
#SBATCH --gpus=1           
#SBATCH --gpus-per-task=1     
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --container-image= ./sbatch/xray.sqsh 
#SBATCH --container-mounts=/purestorage:/purestorage  
#SBATCH --container-writable           
#SBATCH --output=logs/eval.out      
#SBATCH --error=logs/eval.err       

cd X-Ray-Object-Detector

python eval.py