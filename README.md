# Computer Vision [Team 12]
Computer Vision coursework - University of Bath (Team 12)
<br>
<br>
### Image Data
If there is no image data provided in the `/data` folder or you want to try out your own images with the algorithms, please do the following:
<br>
**(Task 1) Angle Finding**
Please put your images into `src/data/task_1/test_images`, and update the `list.txt` file accordingly. The format for the `list.txt` file is [image_name].[file_type (e.g. png)],[angle in degrees] (without the [])
<br>
**(Task 2/Task 3) Template Matching/SIFT**
Please put your test images into `src/data/task_2_3/test_data/images`, the annotations for the test data into `src/data/task_2_3/test_data/annotations` and the training data templates into `src/data/task_2_3/training_data`.
<br>
<br>
### How to Run
There are 2 possible ways how to run the code:<br>
(assuming you are one folder above the `/src` directory)<br>
**1.** Call the `tester.py` via `python3 tester.py`. You will be asked which algorithm to run. The tester will automatically fetch *all* data from the data folder and run the algorithm(s).<br>
**2.** Call the `run_algorithm.py` if you want to run the algorithms with single images. For that, open your console and run the file via `python3 run_algorithm.py <path_to_test_image>`.<br>
<br>
### Settings.ini
Feel free to toggle the parameters in the settings.ini located in the `/src` folder to produce different output/logging/etc. You can also play around with the pyramid parameters that were used for the template matching.