"""
dataloader.py
Helper functions to load and prepare MRI datasets for model training.
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(base_dir, img_size=(224,224), batch_size=32):
    """
    Creates training and validation data generators from directory structure.
    Expects:
        base_dir/
            train/
                class_1/
                class_2/
            val/
                class_1/
                class_2/
    """
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    val_gen = datagen.flow_from_directory(
        os.path.join(base_dir, 'val'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    return train_gen, val_gen
