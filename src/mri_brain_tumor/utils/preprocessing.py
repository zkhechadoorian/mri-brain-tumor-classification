"""
preprocessing.py
Basic preprocessing utilities for MRI data.
"""

import cv2
import numpy as np
import os

def preprocess_image(img_path, size=(224,224)):
    """
    Loads and resizes an MRI image.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img / 255.0
    return img

