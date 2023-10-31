# Computer Vision Tasks
This repository contains three tasks that have been written in Python3 as part of the coursework for the Computer Vision course at the University of Bath.<br>
<br>
The 3 tasks are:
1. Angle Finding
2. Template Matching
3. Scale-Invariant Feature Transform (SIFT)

## Table of Contents
1. [Organizing Image Data](#imagedata)
   1. [Angle Finding](#anglefinder)
   2. [Template Matching & SIFT](#tmsift)
2. [Run Instructions](#howtorun)
3. [Settings](#settings)
4. [License](#license)

## Image Data <a name="imagedata"></a>
If there is no image data provided in the `/data` folder or you want to try out your own images with the algorithms, please do the following:<br>
#### Task 1 - Angle Finding <a name="anglefinder"></a>
Please place your images into `src/data/task_1/test_images`, and update the `list.txt` file accordingly.<br>
The format for the `list.txt` file is `<image_name>.<file_type (e.g., png)>,<angle in degrees>` (without the <>)
<br>
#### Task 2 | Task 3 - Template Matching & SIFT <a name="tmsift"></a>
Please place your test images into `src/data/task_2_3/test_data/images`, the annotations for the test data into `src/data/task_2_3/test_data/annotations` and the training data templates into `src/data/task_2_3/training_data`.

## 2. How to Run <a name="howtorun"></a>
There are 2 possible ways how to run the code:<br>
(assuming you are one folder above the `/src` directory)<br>
**1.** Call the `tester.py` via `python3 tester.py`. You will be asked which algorithm to run. The tester will automatically fetch *all* data from the data folder and run the algorithm(s).<br>
**2.** Call the `run_algorithm.py` if you want to run the algorithms with single images. For that, open your console and run the file via `python3 run_algorithm.py <path_to_test_image>`.

## 3. Settings.ini <a name="settings"></a>
Feel free to toggle the parameters in the settings.ini located in the `/src` folder to produce different output/logging/etc. You can also play around with the pyramid parameters that were used for the template matching.

## 4. License <a name="license"></a>
All code is available under the [MIT License](https://opensource.org/license/mit/). See LICENSE for the full license text.
