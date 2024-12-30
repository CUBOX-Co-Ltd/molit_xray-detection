# Copyright (c) CUBOX, Inc. and its affiliates.

import numpy as np

def calculate_mAP(predictions, ground_truths):
    """Calculates mean Average Precision (mAP)."""
    # Implementation of mAP calculation
    pass

def compute_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area