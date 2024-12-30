# Copyright (c) CUBOX, Inc. and its affiliates.

import cv2
import matplotlib.pyplot as plt

def visualize_predictions(image_path, predictions, output_path="output.jpg"):
    """Visualizes predictions on the image and saves the output."""
    image = cv2.imread(image_path)
    for prediction in predictions:
        x1, y1, x2, y2, conf, class_id = prediction
        label = f"{class_id} ({conf:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(output_path, image)

def plot_metrics(metrics, output_path="metrics.png"):
    """Plots training metrics such as loss, mAP, etc."""
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(output_path)
    plt.show()