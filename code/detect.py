import cv2
import numpy as np
from collections import deque

# Custom imports
from code.mask import mask_hls_colors, mask_region

# Helpers

def _compute_lut_table(gamma):

    table = np.zeros(256)
    for i in np.arange(0, 256):
        table[i] = ((i / 255.0) ** gamma) * 255.0

    table = np.uint8(table)

    return table

def _compute_region_of_interest(n_rows, n_cols, cols_scale = 0.08, rows_scale = 0.62, center_offset = 50):
    """
    Compute the region of interest
    
    Inputs
    ----------
    n_rows: int
        Number of rows in the image/frame
    n_cols: int
        Number of columns in the image/frame
    cols_scale: float
        Scalar determining the proportion of n_cols used in the ROI
    rows_scale: float
        Scalar determining the proportion of n_rows used in the ROI
    center_offset: int
        The number of pixels from the center the two "top" points of the ROI will be found
        
    Outputs
    -------
    region_of_interest: numpy.ndarray
        Numpy array containing four points defining the ROI
    """

    region_of_interest = np.array([[(cols_scale * n_cols, (1 - cols_scale) * n_rows), 
                                    ((n_cols // 2) - center_offset, rows_scale * n_rows), 
                                    ((n_cols // 2) + center_offset, rows_scale * n_rows), 
                                    ((1 - cols_scale) * n_cols, (1 - cols_scale) * n_rows)]], dtype = np.int32)
    return region_of_interest

# Lane detector

class LaneDetector:

    def __init__(self, n_rows, n_cols, config, buffer_size = 8):
        
        self.n_rows = n_rows
        self.n_cols = n_cols

        # Gamma correction table
        self.table = _compute_lut_table(config["gamma"])

        # HLS colors
        self.hls_lower1 = np.array(config["hls_white_lower1"])
        self.hls_upper1 = np.array(config["hls_white_upper1"])
        self.hls_lower2 = np.array(config["hls_yellow_lower2"])
        self.hls_upper2 = np.array(config["hls_yellow_upper2"])

        # Gaussian blurring
        self.kernel = config["gaussian_kernel"]

        # Region of interest masking
        self.region = _compute_region_of_interest(n_rows, n_cols)

        # Canny edge detection
        self.canny_low = config["canny_low"]
        self.canny_high = config["canny_high"]

    def detect(self, image):

        gamma_image = cv2.LUT(image, self.table)

        masked_color_image = mask_hls_colors(gamma_image, self.hls_lower1, self.hls_upper1, 
                                                          self.hls_lower2, self.hls_upper2)

        grayscale_image = cv2.cvtColor(masked_color_image, cv2.COLOR_BGR2GRAY)

        blurred_image = cv2.GaussianBlur(grayscale_image, (self.kernel, self.kernel), 0)

        edges_image = cv2.Canny(blurred_image, self.canny_low, self.canny_high)

        masked_edges = mask_region(edges_image, self.region)

        return masked_edges
