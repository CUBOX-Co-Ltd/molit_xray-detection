# Copyright (c) CUBOX, Inc. and its affiliates.

import argparse
import cv2
import yaml
from pathlib import Path

def load_class_names(config_path):
    """Load class names from the config YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['names']

def draw_ground_truth(image, label_path, class_names):
    """Draw ground truth boxes on the image based on YOLO label file."""
    with open(label_path, 'r') as f:
        labels = f.readlines()

    height, width, _ = image.shape 

    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        w = float(parts[3]) * width
        h = float(parts[4]) * height

        # Convert from YOLO format (center, width, height) to (x1, y1, x2, y2)
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for GT

        # Use class names from the config file
        class_name = class_names[class_id]
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

def inference(input_folder, label_folder, output_folder, class_names):
    input_folder = Path(input_folder)
    label_folder = Path(label_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for image_path in input_folder.glob('*.*'):
        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:  # Filter image formats
            continue
        
        print(f"Processing {image_path}...")

        image = cv2.imread(str(image_path))
        label_path = label_folder / f"{image_path.stem}.txt"  # Corresponding label file

        if label_path.exists():
            image_with_gt = draw_ground_truth(image, label_path, class_names)
            output_image_path = output_folder / f"{image_path.stem}_gt_labels{image_path.suffix}"
            cv2.imwrite(str(output_image_path), image_with_gt)
            print(f"Saved output with ground truth labels to {output_image_path}")
        else:
            print(f"Warning: No label file found for {image_path.stem}")

def main():
    parser = argparse.ArgumentParser(description="Run inference on a folder of images.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file")
    parser.add_argument('--labels', type=str, required=True, help="Path to the folder containing label files in YOLO format")
    parser.add_argument('--input', type=str, required=True, help="Path to the input image folder")
    parser.add_argument('--output', type=str, required=True, help="Path to the output folder")

    args = parser.parse_args()
    
    class_names = load_class_names(args.config)

    inference(args.input, args.labels, args.output, class_names)

if __name__ == '__main__':
    main()
