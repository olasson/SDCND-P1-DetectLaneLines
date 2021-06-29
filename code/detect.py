import cv2
import numpy as np
from collections import deque

# Custom imports
from code.mask import mask_hls_colors, mask_region
from code.lines import line_slope, line_extend, line_is_lane_line

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

    def __init__(self, n_rows, n_cols, buffer_size = 8):
        
        #self.n_rows = n_rows
        #self.n_cols = n_cols

        # Gamma correction table
        gamma = 2.1
        self.table = _compute_lut_table(gamma)

        # HLS color masking
        self.hls_lower1 = np.array([0, 200, 0])
        self.hls_upper1 = np.array([255, 255, 255])
        self.hls_lower2 = np.array([10, 10, 150])
        self.hls_upper2 = np.array([40, 255, 255])

        # Gaussian blurring
        self.kernel = 5

        # Region of interest masking
        self.region = _compute_region_of_interest(n_rows, n_cols)

        # Canny edge detection
        self.canny_low = 50
        self.canny_high = 150

        # Hough lines
        self.resolution_distance = 1
        self.resolution_angular = np.pi / 180
        self.min_number_of_votes = np.array([10, 10, 10, 10, 20, 30, 40, 50])
        self.max_line_gaps = np.array([100, 20, 40, 60, 80, 100, 200, 300])
        self.min_line_lengths = np.array([5, 10, 30, 40, 50, 60, 70, 80])
        

    def detect(self, image):

        gamma_image = cv2.LUT(image, self.table)

        masked_color_image = mask_hls_colors(gamma_image, self.hls_lower1, self.hls_upper1, 
                                                          self.hls_lower2, self.hls_upper2)

        grayscale_image = cv2.cvtColor(masked_color_image, cv2.COLOR_BGR2GRAY)

        blurred_image = cv2.GaussianBlur(grayscale_image, (self.kernel, self.kernel), 0)

        edges_image = cv2.Canny(blurred_image, self.canny_low, self.canny_high)

        masked_edges = mask_region(edges_image, self.region)

        return masked_edges
