# Copyright (c) CUBOX, Inc. and its affiliates.

import os
import torch
from ultralytics import YOLO

# Set environment variables for GPU configuration
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Ensures that CUDA calls are synchronous for debugging purposes.
os.environ['TORCH_USE_CUDA_DSA'] = "1"   # Enable CUDA device-side assertions for debugging.
os.environ['OMP_NUM_THREADS'] = "1"      # Set the number of OpenMP threads for parallel computation.
os.environ["NCCL_NET"] = "Socket"        # Set the network interface to "Socket" for distributed training.
torch.cuda.empty_cache()                 # Clear the GPU cache to avoid out-of-memory errors.

if __name__ == '__main__':
    model = YOLO("runs/train/yolo11x/weights/best.pt")  # Path to the best weights from training

    results = model.val(
        data="config/config.yaml",  # Path to the YAML configuration file with dataset details
        project="runs/eval",        # Directory where evaluation results will be saved
        name="yolo11x",             # Name of the evaluation experiment
        imgsz=640,                  # Input image size (640x640 pixels)
        device=[0]                  # List of GPUs to use for evaluation (here using GPU 0)
    )
