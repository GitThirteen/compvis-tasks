import cv2
import os
import numpy as np
import configparser as cfgp
import matplotlib.pyplot as plt

config = cfgp.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../settings.INI'))
config = config['TASK2']

def create_gaussian_pyramid(image, rotations, scale_levels): # todo: more params, alternatively: split stuff into own func
    '''
    Creates a gaussian pyramid for a training data image via (gaussian) blurring and subsequent
    subsampling. Furthermore, for a specified amount of rotations, generates additional
    pyramids and adds them to the dictionary (?).
    Parameters
    ----------
    image : matrix
        the image for which the gaussian pyramid(s) will be created
    rotations : int
        the amount of rotations
    scale_levels : int
        the amount of levels a pyramid will be made out of
    Returns
    -------
    pyramid : dictionary
        a dictionary containing the pyramid for the passed in image. The key is in the form <rotation>_<level>
    '''

    pyramid = { }

    (h, w) = image.shape[:2]
    sigma = config.getfloat('GaussianSigma')
    k = 5 #int(2 * np.ceil(3 * sigma) + 1) # based on the length for the 99 percentile of the gaussian pdf

    for rot in range(rotations):
        key_base = '' + str(rot) + '_'

        rot_m = cv2.getRotationMatrix2D((h // 2, w // 2), (360 / rotations) * rot, 1.0)
        img = cv2.warpAffine(image, rot_m, (w, h))
        pyramid[key_base + str(0)] = img

        for lvl in range(scale_levels):
            key = key_base + str(lvl + 1)

            img = cv2.GaussianBlur(img, (k, k), sigma)
            img = np.delete(img, range(1, img.shape[0], 2), axis=0)
            img = np.delete(img, range(1, img.shape[1], 2), axis=1)
            print(img.shape)
            pyramid[key] = img

    if config.getboolean('ShowPyramid') == True:
        _, axarr = plt.subplots(rotations, scale_levels)
        plt.subplots_adjust(wspace=0)

        for rot in range(rotations):
            for lvl in range(scale_levels):
                key = '' + str(rot) + '_' + str(lvl)
                pos = axarr[rot, lvl]
                pos.imshow(pyramid[key])
                pos.set_xticks([])
                pos.set_yticks([])

        plt.show()

    return pyramid


def preprocess(pyramids):
    '''
    Sets the background to 0 (black) for each scaled and rotated template.
    Parameters
    ----------
    param_one : type
        param_desc
    pram_two : type
        param_desc
    param_three : type
        param_desc
    Returns
    -------
    return_val : type
        val_desc
    '''
    # TODO
    return


def template_match(img, template_list):
    '''
    Convolves a set of templates across an image and returns a list containing the bbox of matches.
    code based off tutorial : https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html

    Parameters
    ----------
    img : np.ndarray
     return value of cv2.imread(img_path), a 3d numpy array

    template_list: list
     list of templates to be matched in image: [ndarray_temp_1, ndarray_temp_2,...]. Each template should be a 3D numpy array with using the same axis ordering conventions as used by arrays returned from cv2.imread()
    
    Returns
    -------
    bboxs_list: list
     list containing bbox coordinate tuples for each template identifying location of template in image: [[(x0,y0),(x1,y1)],...] if no match for a template, None tuple pairs returned for coordinates of bbox of that tuple. 
    '''
    # TODO
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
    for template in template_list:
        # extract template dims
        w, h = template.shape[::-1]
        
        # initialise lists to track bbox generated from each method
        top_left_corners = np.zeros((len(methods),2))
        bottom_right_corners = np.zeros((len(methods),2))

        # loop over similarity scores
        for idx,meth in enumerate(methods):
            img2 = img.copy()
            method = eval(meth)
            
            # Apply template Matching
            res = cv2.matchTemplate(img2,template,method)
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
            # cv2.rectangle(img2,top_left, bottom_right, 255, 2)

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



        



def draw(): # todo: more params
    '''
    Draws a bounding box around each detected object, including its class label.
    Parameters
    ----------
    param_one : type
        param_desc
    pram_two : type
        param_desc
    param_three : type
        param_desc
    Returns
    -------
    return_val : type
        val_desc
    '''
    # TODO
    return


def get_images(folder_path):
    valid_types = ['.jpg', '.png']
    collector = [ ]

    for file_name in os.listdir(folder_path):
        _, file_type = os.path.splitext(file_name)
        if file_type.lower() not in valid_types:
            continue

        image = cv2.imread(folder_path + '/' + file_name)
        collector.append(image)

    return collector


def main():
    images = get_images(config.get('TrainingDataPath'))
    rots = config.getint('PyramidRotations')
    scale = config.getint('ScaleLevels')

    pyramids = [ ]

    for image in images:
        pyramid = create_gaussian_pyramid(image, rots, scale)
        break
        pyramids.append(pyramid)

    # TODO Stuff with pyramids

if __name__ == "__main__":
    main()