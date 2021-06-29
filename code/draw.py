"""
This file contains some simple functions for drawing lines
"""

import cv2

import numpy as np

def draw_line(image, line, color = [0, 0, 255], thickness = 5):
    """
    Draw a line in an image
    
    Inputs
    ----------
    image: numpy.ndarray
        Numpy array containing a single RGB image
    color: numpy.ndarray
        Numpy array containing a single RGB color
    thickness: int
        Integer specifying the thickness of 'line'
    Outputs
    -------
        N/A
        
    """
    cv2.line(image, (int(line[0][0]), int(line[0][1])),
                    (int(line[0][2]), int(line[0][3])), color, thickness)

def draw_region(image, region, color = [255, 0, 0], thickness = 5):
    """
    Draw an arbitrary region defined by four points into an image
    
    Inputs
    ----------
    image: numpy.ndarray
        Numpy array containing a single RGB image
    region: numpy.ndarray
        Numpy array four points defining a region
    color: numpy.ndarray
        Numpy array containing a single RGB color
    thickness: int
        Integer specifying the thickness of 'line'
    Outputs
    -------
        N/A
        
    """

    region = np.int32(region)

    cv2.polylines(image, [region], True, color, thickness)