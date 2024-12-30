# Copyright (c) CUBOX, Inc. and its affiliates.

import argparse
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

def inference(model, input_folder, output_folder, save_annot=False):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for image_path in input_folder.glob('*.*'):
        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:  # Filter image formats
            continue
        
        print(f"Processing {image_path}...")

        image = cv2.imread(str(image_path))
        original_image = image.copy()

        results = model(image)

        if results:
            result = results[0]

            annotated_image = result.plot()
            output_image_path = output_folder / f"{image_path.stem}_inference{image_path.suffix}"
            cv2.imwrite(str(output_image_path), annotated_image)
            print(f"Saved output to {output_image_path}")
            
            if save_annot:
                label_file_path = output_folder / f"{image_path.stem}_labels.txt"
                with open(label_file_path, 'w') as f:
                    for box, conf, cls_id, cls_name in zip(result.boxes.xywh, result.boxes.conf, result.boxes.cls, result.names):
                        x, y, w, h = box
                        label_line = f"{cls_name} {conf:.4f} {x} {y} {w} {h}\n"
                        f.write(label_line)
                print(f"Saved annotations to {label_file_path}")
        else:
            print(f"No objects detected in {image_path}")

def main():
    parser = argparse.ArgumentParser(description="Run inference on a folder of images.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input image folder")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--output', type=str, required=True, help="Path to the output folder")
    parser.add_argument('--save_annot', action='store_true', help="Save annotations (bounding boxes and class labels) as .txt files")

    args = parser.parse_args()

    print(f"Loading model from {args.ckpt}...")
    model = YOLO(args.ckpt)

    inference(model, args.input, args.output, args.save_annot)

if __name__ == '__main__':
    main()
