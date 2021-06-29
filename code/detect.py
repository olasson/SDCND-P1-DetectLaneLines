import cv2
import numpy as np
from collections import deque

# Custom imports
from code.mask import mask_hls_colors

# Helpers

def _compute_lut_table(gamma):

    table = np.zeros(256)
    for i in np.arange(0, 256):
        table[i] = ((i / 255.0) ** gamma) * 255.0

    #table = np.uint8(table)

    return table

class LaneDetector:

    def __init__(self, n_rows, n_cols, config, buffer_size = 8):
        
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.table = _compute_lut_table(config["gamma"])

        # HLS colors
        self.hls_lower1 = np.array(config["hls_white_lower1"])
        self.hls_upper1 = np.array(config["hls_white_upper1"])
        self.hls_lower2 = np.array(config["hls_yellow_lower2"])
        self.hls_upper2 = np.array(config["hls_yellow_upper2"])

        self.kernel = config["gaussian_kernel"]


    def detect(self, image):

        gamma_image = cv2.LUT(image, self.table.astype('uint8'))

        masked_color_image = mask_hls_colors(gamma_image, self.hls_lower1, self.hls_upper1, 
                                                          self.hls_lower2, self.hls_upper2)

        grayscale_image = cv2.cvtColor(masked_color_image, cv2.COLOR_BGR2GRAY)

        blurred_image = cv2.GaussianBlur(grayscale_image, (self.kernel, self.kernel), 0)

        return blurred_image
