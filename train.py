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
    model = YOLO(
        model="models/yolo11x.pt",  # Path to the pre-trained model checkpoint (in .pt format)
        task="detect"         # Task type: object detection
    )

    results = model.train(
        data="config/config.yaml",  # Path to the YAML configuration file containing dataset details
        epochs=600,                 # Number of epochs to train the model
        project="runs/train",       # Directory to store training results
        name="yolo11x",             # Name of the experiment or model
        imgsz=640,                  # Input image size (640x640 pixels)
        cache=False,                # Disable image caching for training
        batch=640,                  # Batch size for training
        device=[0, 1, 2, 3, 4, 5, 6, 7],  # List of GPU devices to use for training (0 to 7)
    )

    # Export the trained model in ONNX format for inference in other frameworks
    model.export(format="onnx")
