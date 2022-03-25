import cv2
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import configparser as cfgp
from helpers import draw_gaussian_pyramid, get_images

config = cfgp.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../settings.INI'))
config = config['TASK2']
'''
TODO:
Create a function to take in a list of scales and a gaussian pyramid of a template and return the template at those scales. - Michael
Update template matching and research scores - Vish
'''
def create_gaussian_pyramid(image, rotations, scale_levels):
    """
    Creates a gaussian pyramid for a training data image via blurring and subsequent
    subsampling. Furthermore, for a specified amount of rotations, generates additional
    pyramids and adds them to the dictionary.

    Parameters
    ----------
    image : matrix
        the image for which the gaussian pyramid(s) will be created

    rotations : int
        the amount of rotations

    scale_levels : int
        the amount of subsampled images + the base image

    Returns
    -------
    pyramid : dict
        a dictionary containing the pyramid for the passed in image. The key is the rotation factor (e.g. 0, 1, 2, 3) of the image.
    """

    pyramid = { }

    # fetch height, width of image & sigma and k for gaussian kernel
    (h, w) = image.shape[:2]
    sigma = config.getfloat('GaussianSigma')
    k = config.getint('GaussianKernelSize')

    # rotation step size
    step = 360 / rotations
    
    # loop for every rotation
    for rot in range(rotations):
        # create a rot matrix and rotate image
        rot_m = cv2.getRotationMatrix2D((h // 2, w // 2), step * rot, 1.0)
        img = cv2.warpAffine(image, rot_m, (w, h))
        
        # add first "default" image as pyramid base
        pyramid[rot] = [ img ]
        
        # loop for the rest of the pyramid
        for level in range(1, scale_levels):
            # blur image with gaussian and remove all even rows
            # and cols (subsampling)
            '''
            TODO:
            Research how sigma / gaussian kernel size should change as level increases
            '''
            img = cv2.GaussianBlur(img, (k, k), sigma)
            img = np.delete(img, range(1, img.shape[0], 2), axis=0)
            img = np.delete(img, range(1, img.shape[1], 2), axis=1)
            # add new image to pyramid
            pyramid[rot].append(img)

    if config.getboolean('ShowPyramid') == True:
        draw_gaussian_pyramid(pyramid, scale_levels, rotations)

    return pyramid


def preprocess(pyramid):
    """
    Sets the background to 0 (black) for each scaled and rotated template.

    ASSUMPTION: Assume template image itself does not get affected by this

    Parameters
    ----------
    pyramids : list
        A list of dictionaries representing the gaussian pyramids

    Returns
    -------
    pyramids : list
        Preprocessed list of dictionaries representing the gaussian pyramids
    """
    for rot_key in list(pyramid.keys()):
        scale_levels = pyramid[rot_key]
        for idx, img in enumerate(scale_levels):
            # Set background of img to black
            ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 245, 255, cv2.THRESH_BINARY)
            img[thresh == 255] = 0
            scale_levels[idx] = img
        pyramid[rot_key] = scale_levels
    return pyramid


def get_bbox_dims(img):
    """
    Finds dimensions for bounding boxes in passed in image

    Parameters
    ----------
    img : np.ndarray
     return value of cv2.imread(img_path), a 3d numpy array

    Returns
    -------
    dims_list: list
     list containing width and height of each bbox in image as tuples [(width, height), ....]
    """
    dims_list = []
    new = img.copy()

    # Blur the image
    blur = cv2.GaussianBlur(new, (11, 11), 3)
    # Convert the image to grayscale
    grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary
    ret, binary = cv2.threshold(grey, 250, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(~binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w * h) > 500:
            dims = (w, h)
            cv2.rectangle(new,(x,y), (x+w,y+h), (255,0,0), 5)
            dims_list.append(dims)

    return dims_list


def extract_templates_from_pyramid(pyramid, bboxes, option='closest'):
    """
    TODO Description
    Options: closest, upper, lower, both

    CHECK IF RESAMPLING CAN FIX ISSUES

    Parameters:
    pyramid : dict
        keys: rotation_index , values: list of template arrays at different scales with a given orientation
    
    bboxes : list
        list of (w,h) tuples for the bboxes found of objects in the image
    
    option : str
        closest : returns scales 

    Returns:
    --------
    filtered_pyramid : dict
        keys: rotation_index , values: list of template arrays at required scales with a given orientation

    """

    final_scale_diags = [ ]
    
    first_scale_list = list(pyramid.values())[0]
    bbox_diags = [np.sum(scale_level.shape[:-1]**2)**0.5 for scale_level in first_scale_list]

    def closest():
        closest = min(bbox_diags, key=lambda x: abs(x - diag))
        if closest not in final_scale_diags:
            final_scale_diags.append(closest)

    def upper():
        c_bbox_diags = bbox_diags.copy()
        c_bbox_diags.reverse()

        upper = c_bbox_diags[-1]
        for d in c_bbox_diags:
            if diag < d:
                upper = d
                break

        if upper not in final_scale_diags:
            final_scale_diags.append(upper)

    def lower():
        lower = bbox_diags[-1]
        for d in bbox_diags:
            if diag > d:
                lower = d
                break

        if lower not in final_scale_diags:
            final_scale_diags.append(lower)

    def both():
        upper()
        lower()

    funcs = {
        'closest': closest,
        'upper': upper,
        'lower': lower,
        'both': both
    }

    for bbox in bboxes:
        diag = np.sqrt(bbox[0] ** 2 + bbox[1] ** 2)

        if option not in funcs:
            # In case an invalid option was specified
            print(f'[ERROR] Option {option} does not exist.')
            sys.exit(1)

        funcs[option]()
    
    filtered_pyramid = {}

    indices = [bbox_diags.index(el) for el in final_scale_diags]
    for rot_key, level_list in pyramid.items():
        filtered_pyramid[rot_key] = [scaled_template for scale_level_index, scaled_template in enumerate(level_list) if scale_level_index in indices]

    return result
        

def template_match(img, img_scale, template_dict):
    """
    Convolves a set of templates across an image and returns a list containing the bbox of matches.
    code based off tutorial : https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html

    Parameters
    ----------
    img : np.ndarray
     return value of cv2.imread(img_path), a 3d numpy array

    template_dict: dict
					key - class for template, eg 'theater, silo, flower ...'
					value - list of templates for that class (varied resolution, rotation, scales etc)	
     to be matched in image: [ndarray_temp_1, ndarray_temp_2,...]. Each template should be a 3D numpy array with using the same axis ordering conventions as used by arrays returned from cv2.imread()
    
    Returns
    -------
    bboxs_dict: dict
					key - class of template 
					value - list containing bbox corners for cv2.rectangles [upper_left, bottom_right] 
					[[(x0,y0),(x1,y1)],...] if no match for a template, None tuple pairs returned for coordinates of bbox of that tuple. 
    """
    # TODO
				# IMPLEMENT DICTs
    # test code
    # Figure out way to pick the 'best' result from all the methods results. Define best ? Median bbox values?
    # check how long for loop takes to loop over all templates, might wanna do a 3d convolution instead of several 2d convolves for performance
    # - research what the difference between each method is
    # figure out where the threshold is set for the response - will a 'match' always be found?

    # EDGE CASES:
    # no match found
    # multiple matches found

    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    # init bbox array
    bboxs_list = []

    # loop over templates
    for template in template_dict:
        # extract template dims
        w, h = template.shape[::-1]
        
        # extract template at correct scales
        scaled_templates = extract_templates_from_pyramid(template, scale_list, option="upper")
        
        for temps in scaled_templates:
            # initialise lists to track bbox generated from each method
            top_left_corners = np.zeros((len(methods),2))
            bottom_right_corners = np.zeros((len(methods),2))

            # loop over similarity scores
            for idx,meth in enumerate(methods):
                            img2 = img.copy()
                            method = eval(meth)
                            
                            # Apply template Matching
                            res = cv2.matchTemplate(img2, template, method)
                            '''
                            EDGE CASE: multiple matches of template in image cv2.minMaxLoc will not be sufficient
                            how to detect if multiple detections?
                            '''
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 
                            
                            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                                            top_left = np.asarraymin_loc
                            else:
                                            top_left = np.asarray(max_loc)
                            bottom_right = (top_left[0] + w, top_left[1] + h)

                            # add corners to arrays
                            top_left_corners[idx] = top_left
                            bottom_right_corners[idx] = bottom_right

                            # draw bbox on img for debugging
                            if config.getboolean('ShowBbox') == True:
                                            cv2.rectangle(img2,top_left, bottom_right, 255, 2)

        # find median corners - Introduce debugging print statements here, median taking could give poor results
        '''
        check edge case of no matchs handled properly at this step , should return [(none,none), (none,none)]
        '''
        median_top_left = np.median(top_left_corners,0)
        median_bottom_right = np.median(bottom_right_corners,0)

        median_top_left = [None,None] if (median_bottom_right == 0).all() else median_bottom_right
        median_bottom_right = [None,None] if (median_bottom_right == 0).all() else median_bottom_right

        bboxs_list.append([median_top_left,median_bottom_right])

    return bboxs_list



def draw(img, bboxes):
    '''
    Draws a bounding box around each detected object, including its class label.

    Parameters
    ----------
    img : np.ndarray
        return value of cv2.imread(img_path), a 3d numpy array

    bboxes : dictionary
        dictionary with the image label as the key and the bbox top left and bottom right stored as tuples in a list as the value.
        e.g. {"hydrant": [top_left, bottom_right], ...}

    Returns
    -------
    final_image : np.ndarray
        Final image with bboxes and labels represented as a 3d numpy array
    '''
    result = img.copy()
    for label, bbox in bboxes.items():
        cv2.rectangle(result, bbox[0], bbox[1], (255, 0, 0), 2)
        cv2.putText(result, label, (bbox[0][0], bbox[1][1]+13), 0, 0.5, (255,0,0))

    plt.imshow(result)
    plt.show()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("png_path", help="Path to a test image")
    args = parser.parse_args()

    templates = get_images(config.get('TrainingDataPath'))
    test_image = cv2.imread(args.png_path)
    rots = config.getint('PyramidRotations')
    scale = config.getint('ScaleLevels')

    pyramids = [ ]

    for template in templates:
        pyramid = create_gaussian_pyramid(template, rots, scale)
        pyramids.append(pyramid)

    # TODO
    # Iterate over all test_images to get the bbox (currently we're only doing the first one)
    # Then use one of the settings (or all of them if you feel spicy)
    # to extract the right pyramid level(s) and call the template_match function
    bboxes = get_bbox_dims(test_image)

    templates = [ ]
    for pyramid in pyramids:
        for key, value in pyramid.items():
            for scaled_img in value:
                _, thresh = cv2.threshold(cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY), 245, 255, cv2.THRESH_BINARY)
                scaled_img[thresh == 255] = 0
            templates.append(extract_templates_from_pyramid(pyramid, bboxes))

    print(templates)
    
    for template in templates:
        print(type(template))
    

    #extract_templates_from_pyramid(pyramids[0], bboxes, 'upper')
    #extract_templates_from_pyramid(pyramids[0], bboxes, 'lower')
    #extract_templates_from_pyramid(pyramids[0], bboxes, 'both')

    # TODO Stuff with pyramids

if __name__ == "__main__":
    main()
