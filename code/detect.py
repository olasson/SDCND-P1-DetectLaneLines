import cv2
import numpy as np
from collections import deque

# Custom imports
from code.mask import mask_hls_colors, mask_region
from code.lines import line_slope, line_extend, line_is_lane_line
from code.draw import draw_line

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

def _compute_line_filter_values(n_cols, 
                               left_min_scale = 0.10, left_max_scale = 0.35,
                               right_min_scale = 0.70, right_max_scale = 0.95,
                               line_angle_min = 15.00, line_angle_max = 55.00):

    """
    Compute line filter min/max values
    
    Inputs
    ----------
    n_cols: int
        Number of columns in the image/frame
    left_min_scale/left_max_scale: float/float
        Scalar determining the leftmost min/max allowed values in the x-dir
    right_min_scale/right_max_scale: float/float
        Scalar determining the rightmost min/max allowed values in the x-dir
    line_angle_min/line_angle_max: float/float
        Min/max allowed value of the line angle
        
    Outputs
    -------
    line_filter_values: numpy.ndarray
        Numpy array containing line filter values
    """

    left_min = left_min_scale * n_cols
    left_max = left_max_scale * n_cols
    right_min = right_min_scale * n_cols
    right_max = right_max_scale * n_cols

    line_filter_values = np.array([left_min, left_max, right_min, right_max, line_angle_min, line_angle_max])

    return line_filter_values

# Lane detector

class LaneDetector:

    def __init__(self, n_rows, n_cols, buffer_size = 8):

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

        # Line extension parameters
        self.y_bottom = 1.00 * n_rows
        self.y_top = 0.60 *n_rows

        # Line filter
        self.line_filter_values = _compute_line_filter_values(n_cols)

        # Buffers
        self.lines_left_buffer = deque(maxlen = buffer_size)
        self.lines_right_buffer = deque(maxlen = buffer_size)

    def detect(self, image):

        gamma_image = cv2.LUT(image, self.table)

        masked_color_image = mask_hls_colors(gamma_image, self.hls_lower1, self.hls_upper1, 
                                                          self.hls_lower2, self.hls_upper2)

        grayscale_image = cv2.cvtColor(masked_color_image, cv2.COLOR_BGR2GRAY)

        blurred_image = cv2.GaussianBlur(grayscale_image, (self.kernel, self.kernel), 0)

        edges_image = cv2.Canny(blurred_image, self.canny_low, self.canny_high)

        masked_edges = mask_region(edges_image, self.region)
        
        lines_left = []
        lines_right = []  
        
        for i in range(len(self.min_number_of_votes)):

                lines = cv2.HoughLinesP(masked_edges, self.resolution_distance, self.resolution_angular, 
                                        self.min_number_of_votes[i], np.array([]), self.min_line_lengths[i], self.max_line_gaps[i])

                if lines is not None:
                    for line in lines:

                        extended_line = line_extend(line, self.y_bottom, self.y_top)

                        if line_is_lane_line(extended_line, self.line_filter_values):

                            slope = line_slope(extended_line)

                            if slope < 0:
                                lines_left.append(extended_line)
                            else:
                                lines_right.append(extended_line)

        tmp = np.zeros_like(image)

        line_left = None

        if len(lines_left) > 0:
            line_left = np.mean(lines_left, axis = 0)

            self.lines_left_buffer.append(line_left)

        if len(self.lines_left_buffer) > 0:
            line_left = np.average(self.lines_left_buffer, axis = 0)

        if line_left is not None:
            draw_line(tmp, line_left)

        line_right = None

        if len(lines_right) > 0:
            line_right = np.mean(lines_right, axis = 0)

            self.lines_right_buffer.append(line_right)

        if len(self.lines_right_buffer) > 0:
            line_right = np.average(self.lines_right_buffer, axis = 0)

        if line_left is not None:
            draw_line(tmp, line_right)

        lines_image = cv2.addWeighted(image, 0.8, tmp, 1.0, 0.0)

        return lines_image
