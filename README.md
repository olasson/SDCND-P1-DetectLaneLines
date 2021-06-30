# **Detect Lane Lines** 

*by olasson*

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

*This is a revised version of my lane line detection project.*

## Project overview

The majority of the project code is located in the folder `code`:

* [`detect.py`](https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/code/detect.py)
* [`draw.py`](https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/code/draw.py)
* [`io.py`](https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/code/io.py)
* [`lines.py`](https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/code/lines.py)
* [`mask.py`](https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/code/mask.py)
* [`misc.py`](https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/code/misc.py)
* [`plots.py`](https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/code/plots.py)

The main project script is called [`detect_lane_lines.py`](https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/detect_lane_lines.py).

The results of applying the pipeline to images/videos are found in
* `images/results`
* `videos/results`

The images/gifs shown in this readme are found in 

* `images/readme`
* `videos/readme`

## Command line arguments

The following command line arguments are defined:

### Images

* `--images_in:` Folder path to a set of images. Path to a folder containing more folders containing images is valid.
* `--images_out:` Folder path to where pipeline results from --images_in will be stored.
* `--run:` Run the pipeline on a set of provided by --images_in..
* `--n_max_cols:` The maximum number of columns in the image plot.
* `--show:` Shows a set or subset of images provided by --images_in.

### Videos

* `--video_in:` File path to a video file to run the pipeline on.
* `--video_out:` File path to a where the output video will be saved.
* `--fps:` Framerate for the output video.
* `--frame_size:` The frame size of the output video on the form [n_cols, n_rows].

### Misc

* `--force_save:` If enabled, permits overwriting existing data.

All arguments are optional, providing no arguments will cause nothing to happen. 

## Pipeline description

All pipeline configuration is done in the file [`detect.py`](https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/code/detect.py). I have included helper functions to assist in computing parameters that depend on the image size. All configuration is done directly in the class declaration, or through helper functions.

My pipeline can be divided into three main parts, detailed below. 

### Part 1: Prepare image for line extraction

1. Apply a gamma correction to darken the image. For the pipeline, a value of `gamma = 2.1` was used to darken the image and make the color masking more effective. 
<p align="center">
  <img width="70%" height="70%" src="https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/images/readme/step01_gamma.jpg">
</p>

2. Apply a color mask to the image, looking for white and yellow areas. Separating out white and yellow in RGB space can be difficult, which is why HLS space is used here. The HLS representation of white and yellow in used in this project is 

       ...
       self.hls_lower1 = np.array([0, 200, 0]),     # HLS White
       self.hls_upper1 = np.array([255, 255, 255]), # HSL White
       self.hls_lower2 = np.array([10, 10, 150]),   # HLS Yellow
       self.hls_upper2 = np.array([40, 255, 255]),  # HLS Yellow
       ...


As one can see from the masked image, alot of unnecessary details are ignored due to the gamma correction darkening the image, causing the color mask to ignore those areas.

<p align="center">
  <img width="70%" height="70%" src="https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/images/readme/step02_masked.jpg">
</p>

3. Apply grayscale conversion and blur the image using `kernel = (5, 5)`.

<p align="center">
  <img width="70%" height="70%" src="https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/images/readme/step04_blurred.jpg">
</p>

4. Apply Canny edge detection. The parameters used in the edge detection is

        self.canny_low = 50
        self.canny_high = 150

The exact values of these thresholds came from experimentation, but I kept them at a 3:1 ratio as reccomended by Udacity in the course material.
<p align="center">
  <img width="70%" height="70%" src="https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/images/readme/step05_edges.jpg">
</p>

5. Apply a region of interest mask. I used a simple polygon shape for this purpose, defined by four points found by trial and error. The function  `_compute_region_of_interest()` does by defining the following default arguments

        ...
        cols_scale = 0.08, rows_scale = 0.62, center_offset = 50
        ...

which it then uses to compute the ROI as follows:

    ...
    region_of_interest = np.array([[(cols_scale * n_cols, (1 - cols_scale) * n_rows), 
                                    ((n_cols // 2) - center_offset, rows_scale * n_rows), 
                                    ((n_cols // 2) + center_offset, rows_scale * n_rows), 
                                    ((1 - cols_scale) * n_cols, (1 - cols_scale) * n_rows)]], dtype = np.int32)
    ...

Note that `region_of_interest` can vary since it is a function of `(n_rows, n_cols)`, and the three test videos have different frame sizes. This is why it is not computed inside the pipeline, but instead, once at the start. Applying the ROI to the edges image yields the following

<p align="center">
  <img width="70%" height="70%" src="https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/images/readme/step06_region.jpg">
</p>

### Part 2: Extract, filter and draw lines

The first step is to apply a Hough Transform. The parameters are given by
            
     ...
     self.resolution_distance = 1, # [pixels]
     self.resolution_angular = np.pi / 180, # [rad]
     self.min_number_of_votes = np.array([10, 10, 10, 10, 20, 30, 40, 50])
     self.max_line_gaps = np.array([100, 20, 40, 60, 80, 100, 200, 300])
     self.min_line_lengths = np.array([5, 10, 30, 40, 50, 60, 70, 80])
     ...
        
 A `for-loop` then feeds one value of `min_number_of_votes`, `max_line_gaps` and `min_line_lengths` at a time to the Hough Transform, allowing the pipeline to test different combinations of parameters for every image, increasing the odds of detecting lines. If the line set returned is not empty, they are extended between two new points. 
      
After the extension, the lines are filtered by calling `line_is_lane_line(line, line_filter_values)`, where `line_filter_values` is computed by the helper function `_compute_line_filter_values`. This "filter" puts conditions on the horizontal placement of the line, as well as the angle of the line. These parameters were found through trial and error. 

Next, the accepted lines are sorted into right and left based on their slope like so

        slope = line_ops.slope(extended_line)
        if slope < 0:
            lines_left.append(extended_line)
        else:
            lines_right.append(extended_line)

Finally, the mean of the left and right lines respectively is computed (if any lines were in fact found). They are added to their respective line buffer, and averaged if any values exists in the buffer. This provides both robustness and a smoother result in the video.

## Results

*The video results of the pipeline can be seen in the folder `test_videos_output`. Gifs versions of those videos are seen here.*

<h3 align="center">Solid Yellow Right</h3>
</header>

<p align="center">
  <img width="70%" height="70%" src="https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/videos/readme/solidWhiteRight.gif">
</p>

<h3 align="center">Solid Yellow Left</h3>
</header>

<p align="center">
  <img width="70%" height="70%" src="https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/videos/readme/solidYellowLeft.gif">
</p>

<h3 align="center">Challenge</h3>
</header>

<p align="center">
  <img width="70%" height="70%" src="https://github.com/olasson/SDCND-P1-DetectLaneLines/blob/master/videos/readme/challenge.gif">
</p>

## Pipeline Shortcomings and Possible Improvements

The pipeline would likely fail under varying lighting conditions, for example during nighttime or under very harsh lighting. A possible remedy would be a more dynamic tuning of the gamma correction. 

The pipeline could fail if there were crosswalks on the road (as they are white). The line filter would problably have to be tuned and/or better region of interest masking. This could also solve issues that arise when there are lots of cars on the road. 

The pipeline only draws straight lines. In the challenge video, this shows as the lines clearly only partially follows curved lane lines. For the same reason, the pipeline would likely fail on a video with steep turns. 

