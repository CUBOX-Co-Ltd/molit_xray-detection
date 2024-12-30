# Copyright (c) CUBOX, Inc. and its affiliates.

import os
import shutil

def create_folder(path):
    """Creates a folder if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def split_dataset(source_dir, train_dir, val_dir, test_dir, train_ratio=0.8, val_ratio=0.1):
    """Splits dataset into train, val, and test sets."""
    files = os.listdir(source_dir)
    total_files = len(files)
    train_files = int(total_files * train_ratio)
    val_files = int(total_files * val_ratio)

    for i, file in enumerate(files):
        source = os.path.join(source_dir, file)
        if i < train_files:
            shutil.move(source, os.path.join(train_dir, file))
        elif i < train_files + val_files:
            shutil.move(source, os.path.join(val_dir, file))
        else:
            shutil.move(source, os.path.join(test_dir, file))