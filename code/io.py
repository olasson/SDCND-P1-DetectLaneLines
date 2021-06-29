"""
This file contains save and load (I/O) functions.
"""

import cv2
import json
import pickle
import numpy as np
from glob import glob
from os.path import join as path_join

# Custom imports
from code.misc import parse_file_path

# Wrappers

def save_image(file_path, image):

    cv2.imwrite(file_path, image)

    return image

def load_image(file_path):

    image = cv2.imread(file_path)

    return image

# Internals

def _load_images(file_paths):
    """
    Load a set of images into memory
    Inputs
    ----------
    file_paths : list or numpy.ndarray
        A list or array of file_paths - ['./example/myimg1.jpg'... './example/myimgN.jpg',]
    Outputs
    -------
    images: numpy.ndarray
        Array containing 'images'
    file_names: numpy.ndarray
        Array containing the file names - ['myimg1.jpg'... 'myimgN.jpg',]
    """

    n_images = len(file_paths)

    image_shape = load_image(file_paths[0]).shape

    n_rows = image_shape[0]
    n_cols = image_shape[1]

    # RGB or grayscale
    if len(image_shape) > 2:
        n_channels = 3
    else:
        n_channels = 1

    images = np.zeros((n_images, n_rows, n_cols, n_channels), dtype = np.uint8)
    file_names = np.zeros((n_images), dtype = 'U25')

    for i in range(n_images):
        images[i] = load_image(file_paths[i])
        file_names[i] = parse_file_path(file_paths[i])[1]

    return images, file_names

# Glob

def glob_file_paths(folder_path, n_max_samples = None, file_ext = '.png'):
    """
    Load a set of file paths from a folder or nested folders.
    
    Inputs
    ----------
    folder_path: str
        Path to a folder.
    n_max_samples: (None | int)
        The maximum number of file paths that can be loaded.
        If set to None, every file path found in 'folder_path' will be loaded.
    file_ext: str
        Expected file extension of files in 'folder_path'.
       
    Outputs
    -------
    file_paths: numpy.ndarray
        Numpy array containing the file paths matching with 'file_ext' in 'folder_path'.
        
    """

    pattern = path_join(folder_path, '**', '*' + file_ext)

    file_paths = glob(pattern, recursive = True)

    n_samples = len(file_paths)

    if n_max_samples is not None and (n_samples > n_max_samples):
        print('INFO:glob_file_paths(): Picking out ' + str(n_max_samples) + ' random samples from a total set of ' + str(n_samples) + ' samples!')
        
        file_paths = random.sample(file_paths, n_max_samples)

    return file_paths

def glob_images(folder_path, n_max_images = 50):
    """
    Load a set of file paths from a folder or nested folders.
    
    Inputs
    ----------
    folder_path: str
        Path to a folder.
    n_max_samples: (None | int)
        The maximum number of file paths that can be loaded.
        If set to None, every file path found in 'folder_path' will be loaded.
    file_ext: str
        Expected file extension of files in 'folder_path'.
       
    Outputs
    -------
    images: numpy.ndarray
        Numpy array containing the images found in 'folder_path'.
    file_names: numpy.ndarray
        Numpy array containing the file names (not full paths) of the file paths found ing 'folder_path.
        
    """
    file_paths = glob_file_paths(folder_path, n_max_samples = 50, file_ext = '.png')

    if len(file_paths) == 0:
        file_paths = glob_file_paths(folder_path, n_max_samples = 50, file_ext = '.jpg')

    images, file_names = _load_images(file_paths)

    return images, file_names





