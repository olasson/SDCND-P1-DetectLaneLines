"""
This file contains the implementation of the lane detector
"""

import cv2
import numpy as np
from collections import deque

# Custom imports
from code.mask import mask_hls_colors, mask_region
from code.lines import line_slope, line_extend, line_is_lane_line
from code.draw import draw_line

# Helpers

def _compute_lut_table(gamma):
    """
    Compute LUT table for gamma correction
    
    Inputs
    ----------
    gamma: int
        Gamma factor. Higher values leads to darker image and vice versa.

        
    Outputs
    -------
    table: numpy.ndarray
        Numpy array containing the pre-computed lut table.
    """ 

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

    """
    Class for handling lane detection.
    
    Class fields
    ----------
    table: numpy.ndarray
        Numpy array containing the pre-computed lut table.
        Computed by setting 'gamma'
    hls_lower1, hls_upper1: numpy.ndarray, numpy.ndarray
        Lower/upper bound on color 1 in hls space
    hls_lower2, hls_upper2: numpy.ndarray, numpy.ndarray
        Lower/upper bound on color 2 in hls space
    kernel: int
        The kernel size used in gaussian image blurring.
    region: numpy.ndarray
        A numpy array containing four points defining the ROI.
    line_filter_values: numpy.ndarray
        A numpy array containing six values defining what is accepted as a lane line. 
    line_extension_y_values: numpy.ndarray
        A numpy array containing two y-values for extending detected lane lines.
    canny_low/canny_high: int/int
        Threshold values for canny edge detection.
    resolution_distance: int
        The distance resolution in pixels for the Hough accumulator.
    resolution_angular: int
        The angular resolution in radians for the Hough accumulator.
    min_number_of_votes: numpy.ndarray
        A numpy array containing values specifying the min number of votes used by the Hough Transform
    max_line_gaps: numpy.ndarray
        A numpy array containing values specifying the max permitted line gap used by the Hough Transform
    min_line_lengths: numpy.ndarray
        A numpy array containing values specifying the min permitted line length used by the Hough Transform
    lines_left_buffer/lines_right_buffer: numpy.ndarray, numpy.ndarray
        Numpy arrays for buffering up to 'buffer_size' number of previously detected lanes
        
    Class methods
    ----------
    detect(image): numpy.ndarray
        Takes in an image and returns an image with detected lane lines drawn in.

    -------
    """

    def __init__(self, n_rows, n_cols, buffer_size = 8):
        """
        Compute line filter min/max values
        
        Inputs
        ----------
        n_rows/n_cols: int/int
            Number of rows and columns in image or frame
        buffer_size: int
            The number of previous frames kept for buffering
            
        Outputs
        -------
            N/A
        """

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
        self.min_number_of_votes = np.array([10, 10, 10, 10, 20, 30, 40, 50]) # Same length
        self.max_line_gaps = np.array([100, 20, 40, 60, 80, 100, 200, 300]) # Same length
        self.min_line_lengths = np.array([5, 10, 30, 40, 50, 60, 70, 80]) # Same length

        # Line extension parameters
        self.y_bottom = 1.00 * n_rows
        self.y_top = 0.60 *n_rows

        # Line filter
        self.line_filter_values = _compute_line_filter_values(n_cols)

        # Buffers
        self.lines_left_buffer = deque(maxlen = buffer_size)
        self.lines_right_buffer = deque(maxlen = buffer_size)

    def detect(self, image):
        """
        Detect lane lines in an image
        
        Inputs
        ----------
        image: numpy.ndarray
            A single BGR image
            
        Outputs
        -------
        lines_image: numpy.ndarray
            A single BGR image with lane lines drawn in
            
        """

        # Part 1: Pre-process image

        gamma_image = cv2.LUT(image, self.table)

        masked_color_image = mask_hls_colors(gamma_image, self.hls_lower1, self.hls_upper1, 
                                                          self.hls_lower2, self.hls_upper2)

        grayscale_image = cv2.cvtColor(masked_color_image, cv2.COLOR_BGR2GRAY)

        blurred_image = cv2.GaussianBlur(grayscale_image, (self.kernel, self.kernel), 0)

        edges_image = cv2.Canny(blurred_image, self.canny_low, self.canny_high)

        masked_edges = mask_region(edges_image, self.region)
        
        lines_left = []
        lines_right = []  

        # Part 2: Detect lines
        
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

        # Part 3: Post-processing of lines

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
