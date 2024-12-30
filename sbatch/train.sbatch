#!/bin/bash

#SBATCH --partition=hpe
#SBATCH --nodelist=hpe162
#SBATCH --nodes=1                      
#SBATCH --job-name=X-ray_Detector
#SBATCH --ntasks=1                    
#SBATCH --ntasks-per-node=1        
#SBATCH --gpus=8            
#SBATCH --gpus-per-task=8      
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=16G
#SBATCH --container-image= ./sbatch/xray.sqsh 
#SBATCH --container-mounts=/purestorage:/purestorage  
#SBATCH --container-writable           
#SBATCH --output=logs/train.out      
#SBATCH --error=logs/train.err       

cd X-Ray-Object-Detector

python -m torch.distributed.run --nproc_per_node 8 train.py --data configs/config.yaml --weights yolo11x.pt  --device 0,1,2,3,4,5,6,7