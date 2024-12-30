# Copyright (c) CUBOX, Inc. and its affiliates.

from ultralytics import YOLO

def load_model(model_path):
    """Loads the YOLO model from the given path."""
    model = YOLO(model_path)
    return model

def save_model(model, export_path="models/yolo.onnx", format="onnx"):
    """Exports the YOLO model to the specified format."""
    model.export(format=format, path=export_path)