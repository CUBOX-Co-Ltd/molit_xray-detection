# Copyright (c) CUBOX, Inc. and its affiliates.

import logging

def setup_logger(log_file='training.log'):
    """Sets up a logger to track progress."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger()

def log_training_results(logger, results):
    """Logs the training results."""
    logger.info("Training results:")
    for key, value in results.items():
        logger.info(f"{key}: {value}")