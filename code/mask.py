"""
This file contains some basic image operations used in the pipeline. 
"""

import cv2
import numpy as np

def mask_hls_colors(image, hls_lower1, hls_upper1, hls_lower2, hls_upper2):
    """
    Mask two colors in an image
    
    Inputs
    ----------
    image : numpy.ndarray
        Numpy array containing a single RGB image
    hls_lower1, hls_upper1: numpy.ndarray, numpy.ndarray
        Lower/upper bound on color 1 in hls space
    hls_lower2, hls_upper2: numpy.ndarray, numpy.ndarray
        Lower/upper bound on color 2 in hls space
        
    Outputs
    -------
    masked_image: numpy.ndarray
        Image with color mask applied
        
    """

    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
    hls_color1 = cv2.inRange(hls_image, hls_lower1, hls_upper1)
    hls_color2 = cv2.inRange(hls_image, hls_lower2, hls_upper2)
    
    mask = cv2.bitwise_or(hls_color1, hls_color2)
    
    masked_image = cv2.bitwise_and(image, image, mask = mask)
    
    return masked_image
