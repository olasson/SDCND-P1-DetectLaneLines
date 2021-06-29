import cv2
import numpy as np
from collections import deque

# Custom imports
from code.process import mask_hls_colors

# Helpers

def _compute_lut_table(gamma):

    table = np.zeros(256)
    for i in np.arange(0, 256):
        table[i] = ((i / 255.0) ** gamma) * 255.0

    table = np.uint8(table)

    return table

class LaneDetector:

    def __init__(self, n_rows, n_cols, config, buffer_size = 8):
        
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.table = _compute_lut_table(config["gamma"])


    def detect(self, image):

        gamma_image = cv2.LUT(image, self.table)

        return gamma_image
