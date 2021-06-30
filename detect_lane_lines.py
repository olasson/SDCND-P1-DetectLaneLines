"""
This file contains the main script
"""

import cv2
import numpy as np
import argparse
from os.path import join as path_join

# Custom imports
from code.misc import file_exists, folder_guard, folder_is_empty, parse_file_path
from code.io import load_config, glob_images, save_image
from code.plots import plot_images
from code.detect import LaneDetector

FOLDER_DATA = './data'

INFO_PREFIX = 'INFO_MAIN: '
WARNING_PREFIX = 'WARNING_MAIN: '
ERROR_PREFIX = 'ERROR_MAIN: '

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = 'Detect Lane Lines')

    # Images

    parser.add_argument(
        '--images_in',
        type = str,
        nargs = '?',
        default = '',
        help = 'Folder path to a set of images. Path to a folder containing more folders containing images is valid.',
    )

    parser.add_argument(
        '--images_out',
        type = str,
        nargs = '?',
        default = './images/results',
        help = 'Folder path to where pipeline results from --images_in will be stored.'
    )

    parser.add_argument(
        '--run',
        action = 'store_true',
        help = 'Run the pipeline on a set of provided by --images_in.'
    )

    parser.add_argument(
        '--show',
        action = 'store_true',
        help = 'Shows a set or subset of images provided by --images_in.'
    )

    parser.add_argument(
        '--n_max_cols',
        type = int,
        default = 3,
        help = 'The maximum number of columns in the image plot.'
    )

    # Video

    parser.add_argument(
        '--video_in',
        type = str,
        nargs = '?',
        default = '',
        help = 'File path to a video file to run the pipeline on.',
    )

    parser.add_argument(
        '--video_out',
        type = str,
        nargs = '?',
        default = './data/video_output.mp4',
        help = 'File path to a where the output video will be stored.',
    )

    parser.add_argument(
        '--frame_size',
        type = int,
        nargs = '+',
        default = [1280, 720],
        help = 'The frame size of the output video on the form [n_cols, n_rows]'
    )

    parser.add_argument(
        '--fps',
        type = int,
        default = 25,
        help = 'The fps of the output video'
    )

    # Misc

    parser.add_argument(
        '--force_save',
        action = 'store_true',
        help = 'If enabled, permits overwriting existing data.'
    )

    args = parser.parse_args()

    # Init paths

    folder_path_images_in = args.images_in
    folder_path_images_out = args.images_out

    file_path_video_input = args.video_in
    file_path_video_output = args.video_out

    # Init values

    n_max_cols = args.n_max_cols

    fps = args.fps

    # Init flags

    flag_run_on_images = args.run
    flag_show_images = args.show

    flag_run_on_video = (file_path_video_input != '')

    flag_force_save = args.force_save

    # Folder setup

    folder_guard(FOLDER_DATA)

    # Show images

    if flag_show_images:

        if folder_is_empty(folder_path_images_in):
            print(ERROR_PREFIX + 'You are trying to show a set of images but ' + folder_path_images_in + ' is empty or does not exist!')
            exit()

        print(INFO_PREFIX + 'Showing images from folder: ' + folder_path_images_in)

        images, file_names = glob_images(folder_path_images_in)

        plot_images(images, file_names, title_fig_window = folder_path_images_in, n_max_cols = n_max_cols)

        exit()

    # Run on images

    if flag_run_on_images:

        if folder_is_empty(folder_path_images_in):
            print(ERROR_PREFIX + 'You are trying to run the pipeline on a set of images but ' + folder_path_images_in + ' is empty or does not exist!')
            exit()

        if not folder_is_empty(folder_path_images_out):
            print(WARNING_PREFIX + 'The folder ' + folder_path_images_out + ' is not empty! Use --force_save to overwrite its contents.')
            exit()

        print(INFO_PREFIX + 'Running pipeline on images in folder: ' + folder_path_images_in)

        images, file_names = glob_images(folder_path_images_in)

        n_rows, n_cols, n_channels = images[0].shape

        n_images = len(images)

        # Grayscale
        images_results = np.zeros((n_images, n_rows, n_cols, n_channels), dtype = np.uint8)

        lane_detector = LaneDetector(n_rows, n_cols, buffer_size = 0)

        for i in range(n_images):   

            images_results[i] = lane_detector.detect(images[i])

            save_image(path_join(folder_path_images_out, file_names[i]), images_results[i])

        plot_images(images_results, file_names, title_fig_window = folder_path_images_in, n_max_cols = n_max_cols)
    
    # Run on video

    if flag_run_on_video:

        if file_exists(file_path_video_output) and (not flag_force_save):
            print(WARNING_PREFIX + 'The file ' + file_path_video_output + ' already exists! Use --force_save to overwrite it!')
            exit()

        print(INFO_PREFIX + 'Running pipeline on video: ' + file_path_video_input)

        n_rows = args.frame_size[1]
        n_cols = args.frame_size[0]

        cap = cv2.VideoCapture(file_path_video_input)

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path_video_output, fourcc, fps, tuple(args.frame_size))

        lane_detector = LaneDetector(n_rows, n_cols)
        
        i = 0

        while(cap.isOpened()):

            ret, frame = cap.read()

            if ret:
                processed_frame = lane_detector.detect(frame)

                i = i + 1
                if i % 10 == 0:
                    print(INFO_PREFIX + 'Frame ' + str(i) + '/' + str(n_frames))
                
                out.write(processed_frame)
            else:
                break

        cap.release()
        out.release()

        print(INFO_PREFIX + 'Frame ' + str(i) + '/' + str(n_frames))
        print('Done processing video!')
        print('Number of frames successfully processed: ', i)
        print('Result is found here: ', file_path_video_output)








