"""
This file contains some basic line operations used in the pipeline. 
"""

import numpy as np

def line_slope(line):
    """
    Compute line slope

    Inputs
    ----------
    line : numpy.ndarray
        Numpy array containing a single line - [[x1, y1, x2, y2]]
    Outputs
    -------
    slope: float
        Slope of 'line'
    """

    delta_x = float(line[0][2] - line[0][0])
    delta_y = float(line[0][3] - line[0][1])

    if (np.abs(delta_x) < 1e-6) or (np.abs(delta_y) < 1e-6):
        slope = 1e-6
    else:
        slope = delta_y / delta_x

    return slope

def line_angle(line):

    """
    Compute line angle, relative to the horizontal

    Inputs
    ----------
    line : numpy.ndarray
        Numpy array containing a single line - [[x1, y1, x2, y2]]
    Outputs
    -------
    theta: float
        Angle of 'line'
    """   

    slope = line_slope(line)

    theta = np.abs(np.arctan(slope)) * (180 / np.pi)

    return theta

def line_solve_for_x(line, y):
    """
    If y is known, solve for x
    
    Inputs
    ----------
    line : numpy.ndarray
        Numpy array containing a single line - [[x1, y1, x2, y2]]
    y: float
        Known y-coordinate of the point (x,y) which lies on 'line'
        
    Outputs
    -------
    x: float
        Coordinate such that the point (x,y) is found on 'line'
        
    """

    slope = line_slope(line)

    x = ((1 / slope) * (y - line[0][1])) + line[0][0]

    return x


def line_extend(line, y_bottom, y_top):
    """
    Extend line by finding new points

    Inputs
    ----------
    line : numpy.ndarray
        Numpy array containing a single line - [[x1, y1, x2, y2]]
    y_bottom, y_top: float, float
        Known y-coordinates of the extended line
    Outputs
    -------
    line_extended: numpy.ndarray
        Numpy array containing an extended line
    """   

    x_bottom = line_solve_for_x(line, y_bottom)
    x_top = line_solve_for_x(line, y_top)

    line_extended = np.array([[x_bottom, y_bottom, x_top, y_top]])

    return line_extended

def line_is_lane_line(line, line_filter_values):
    """
    Check if line is acceptable as a lane line
    
    Inputs
    ----------
    line_filter_values : numpy.ndarray
        Numpy array containing 3 pairs of min/max filter values
        See _compute_line_filter_values() in detect.py.
        
    Outputs
    -------
    is_lane_line: bool
        'True' if line is acceptable, 'False' otherwise
        
    """

    horizontal_filter = ((line_filter_values[0] <= line[0][0] <= line_filter_values[1]) or
                         (line_filter_values[2] <= line[0][0] <= line_filter_values[3]))

    angle_filter = (line_filter_values[4] <= line_angle(line) <= line_filter_values[5])

    is_lane_line = horizontal_filter and angle_filter

    return is_lane_line