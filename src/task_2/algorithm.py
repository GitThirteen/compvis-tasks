import cv2
import os
import sys
import argparse
import re
import numpy as np
from ..util import Algorithm, Logger, draw_gaussian_pyramid, get_images, get_bbox_dims, get_bbox_iou, config
from icecream import ic

cfg = config(Algorithm.TEMPLATE_MATCHING)
LOGGER = Logger.get()


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

    pyramid = {}

    # fetch height, width of image & sigma and k for gaussian kernel
    (h, w) = image.shape[:2]
    sigma = cfg.getfloat('GaussianSigma')
    k = cfg.getint('GaussianKernelSize')

    # rotation step size
    step = 360 / rotations

    # loop for every rotation
    for rot in range(rotations):
        # create a rot matrix and rotate image
        rot_m = cv2.getRotationMatrix2D((h // 2, w // 2), step * rot, 1.0)
        img = cv2.warpAffine(image, rot_m, (w, h))

        # add first "default" image as pyramid base
        pyramid[rot] = [img]

        # loop for the rest of the pyramid
        for _ in range(1, scale_levels):
            # blur image with gaussian and remove all even rows and cols (subsampling)
            img = cv2.GaussianBlur(img, (k, k), sigma)
            img = np.delete(img, range(1, img.shape[0], 2), axis=0)
            img = np.delete(img, range(1, img.shape[1], 2), axis=1)
            # add new image to pyramid
            pyramid[rot].append(img)

    if cfg.getboolean('ShowPyramid'):
        draw_gaussian_pyramid(pyramid, scale_levels, rotations)

    return pyramid


def generate_pyramids(training_data_path):
    templates = get_images(training_data_path)

    rots = cfg.getint('PyramidRotations')
    scale = cfg.getint('PyramidLevels')

    # Generates a pyramid for all templates
    pyramids = [ ]

    for template in templates:
        pyramid = create_gaussian_pyramid(template, rots, scale)
        pyramids.append(pyramid)

    return pyramids


def preprocess(pyramid):
    """
	Sets the background to 0 (black) for each scaled and rotated template.
	We assume that the object on the template itself does not get affected by this.

	Parameters
	----------
	pyramids : list
	    A list of dictionaries representing the gaussian pyramids

	Returns
	-------
	pyramids : list
		Preprocessed list of dictionaries representing the gaussian pyramids
	"""
    kernel = np.ones(5, 5, np.uint8)

    for rotation in list(pyramid.keys()):
        scale_levels = pyramid[rotation]
        for i, img in enumerate(scale_levels):
            # Set bg of image to black
            _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 245, 255, cv2.THRESH_BINARY)
            img[thresh == 255] = 0
            # Morphologically open image with set kernel to get rid of single pixels
            # that didn't get filtered out in the step before
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

            scale_levels[i] = img
        pyramid[rotation] = scale_levels
    return pyramid

def extract_templates_from_pyramid(pyramid, bboxes, option='closest'):
    """
	Extracts 
	Options: closest, upper, lower, both

	Parameters:
	pyramid : dict
		keys: rotation_index , values: list of template arrays at different scales with a given orientation

	bboxes : list
		list of (w,h) tuples for the bboxes found of objects in the image

	option : str
		closest : picks index closest to diagonal
        upper : picks index with diag length above the one of the diagonal
        lower : picks index with diag length below the one of the diagonal
        both : upper & lower

	Returns:
	--------
	filtered_pyramid : dict
		keys: rotation_index , values: list of template arrays at required scales with a given orientation

	"""

    final_scale_diags = []

    first_scale_list = list(pyramid.values())[0]
    scale_diags = [np.sqrt(scale_lvl.shape[0] ** 2 + scale_lvl.shape[1] ** 2) for scale_lvl in first_scale_list]

    def closest():
        closest = min(scale_diags, key=lambda x: abs(x - diag))
        if closest not in final_scale_diags:
            final_scale_diags.append(closest)

    def upper():
        c_bbox_diags = scale_diags.copy()
        c_bbox_diags.reverse()

        upper = c_bbox_diags[-1]
        for d in c_bbox_diags:
            if diag < d:
                upper = d
                break

        if upper not in final_scale_diags:
            final_scale_diags.append(upper)

    def lower():
        lower = scale_diags[-1]
        for d in scale_diags:
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
            LOGGER.ERROR(f'Option {option} does not exist.')
            sys.exit(1)

        funcs[option]()

    filtered_pyramid = {}
    indices = [scale_diags.index(el) for el in final_scale_diags]

    for rot_key, level_list in pyramid.items():
        filtered_pyramid[rot_key] = [scaled_template for scale_level_index, scaled_template in enumerate(level_list) if
                                     scale_level_index in indices]

    return filtered_pyramid


def template_match(img, N, template_dict):
    """
	Convolves a set of templates across an image and returns a list containing the bbox of matches.
	code based off tutorial : https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html

	Parameters
	----------
	img : np.ndarray
		return value of cv2.imread(img_path), a 3d numpy array

	N : int
		number of objects found in test image

	template_dict: dict
					key - class for template, eg 'theater, silo, flower ...'
					value - pyramid dict of rotated templates for that class (k: rotation_idx, v: list of scaled templates)
		Each template is 3D numpy array with using the same axis ordering conventions as used by arrays returned from cv2.imread()

	Returns
	-------
	results_dict: dict
					key - class of template
					value - list containing bbox corners for cv2.rectangles [upper_left, bottom_right]

	"""
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    # 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    methods = ['cv2.TM_SQDIFF']
    # init results dict
    class_bbox_scores_dict = {}

    # loop over classes
    for class_, template_pyramid in template_dict.items():
        # class_bboxes = []
        # initialise score dict to keep track of template giving highest score
        class_scores_dict = {}

        # loop over orientations
        for template_idx, (rot_key, rotated_scaled_templates) in enumerate(template_pyramid.items()):
            # loop over scaled templates
            for template in rotated_scaled_templates:
                # extract template dims
                w, h = template.shape[0], template.shape[1]
                # loop over similarity scores
                for idx, meth in enumerate(methods):
                    img2 = img.copy()
                    method = eval(meth)

                    # Apply template Matching
                    res = cv2.matchTemplate(img2, template, method)
                    '''
					EDGE CASE: multiple matches of template in image cv2.minMaxLoc will not be sufficient
					how to detect if multiple detections?
					'''
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    # update score dict if current template has a better similarity / match
                    if template_idx == 0:
                        # initialise class score dict if first iteration in template loop
                        if meth in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']:
                            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                            class_scores_dict[meth] = {'score': min_val}
                            top_left = np.asarray(min_loc)
                            bottom_right = np.asarray((top_left[0] + w, top_left[1] + h))
                            class_scores_dict[meth]['bbox'] = (top_left, bottom_right)

                        else:
                            class_scores_dict[meth] = {'score': min_val}
                            top_left = np.asarray(max_loc)
                            bottom_right = np.asarray((top_left[0] + w, top_left[1] + h))
                            class_scores_dict[meth]['bbox'] = (top_left, bottom_right)

                    else:
                        # only update dict if scores are better
                        if meth in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']:
                            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                            if class_scores_dict[meth]['score'] > min_val:
                                class_scores_dict[meth]['score'] = min_val
                                top_left = np.asarray(min_loc)
                                bottom_right = np.asarray((top_left[0] + w, top_left[1] + h))
                                class_scores_dict[meth]['bbox'] = (top_left, bottom_right)
                        else:
                            if class_scores_dict[meth]['score'] < max_val:
                                class_scores_dict[meth]['score'] = max_val
                                top_left = np.asarray(max_loc)
                                bottom_right = np.asarray((top_left[0] + w, top_left[1] + h))
                                class_scores_dict[meth]['bbox'] = (top_left, bottom_right)

                # LOGGER.DEBUG_IC(class_,class_scores_dict[meth]['score']/1e6)
                # cv2.rectangle(img2,top_left,bottom_right,(255, 0, 0), 2)
                # cv2.imshow('bbox found',img2)
                # cv2.imshow('temp',template)
                # cv2.waitKey(0)
        # store final results for class in dict
        class_bbox_scores_dict[class_] = class_scores_dict

    # Check if any classes have overlapping bboxes, and remove class with highest score if overlapping
    classes_to_remove = []
    for class_, class_scores_dict in class_bbox_scores_dict.items():
        for class2_, class_scores_dict2 in class_bbox_scores_dict.items():
            if class_ == class2_:
                continue

            iou = get_bbox_iou(class_scores_dict['cv2.TM_SQDIFF']['bbox'], class_scores_dict2['cv2.TM_SQDIFF']['bbox'])

            if iou > 0.3:
                if class_scores_dict['cv2.TM_SQDIFF']['score'] < class_scores_dict2['cv2.TM_SQDIFF']['score']:
                    if class2_ not in classes_to_remove:
                        classes_to_remove.append(class2_)
                else:
                    if class_ not in classes_to_remove:
                        classes_to_remove.append(class_)

    for class_ in classes_to_remove:
        del class_bbox_scores_dict[class_]

    # pick a N distinct bboxs, where N is number of objects found in image
    classes = list(class_bbox_scores_dict.keys())
    sorted_results = np.zeros((len(classes), len(methods)))

    # loop over methods
    for idx, meth in enumerate(methods):
        method_scores = []
        # for each class look at the similarity scores given using that method
        for class_, class_scores_dict in class_bbox_scores_dict.items():
            method_scores.append((class_, class_scores_dict[meth]['score'] / 1e6))
        # sort scores to find classes that were most strongly picked up by each method
        sorted_method_scores = sorted(method_scores, key=lambda x: x[1])
        # sort in descending order if meth TM_SQDIFF or TM_SQDIFF_NORMED else ascending
        sorted_method_scores = sorted_method_scores[::-1] if meth in ["TM_SQDIFF",
                                                                      "TM_SQDIFF_NORMED"] else sorted_method_scores
        sorted_results[:, idx] = [classes.index(class_score_tuple[0]) for class_score_tuple in sorted_method_scores]

    # pick top N classes
    top_n_classes = []
    for class_idx in np.median(sorted_results, 1):
        class_ = classes[int(class_idx)]
        if class_ not in top_n_classes and len(top_n_classes) < N: top_n_classes.append(class_)

    # for each class find the median bbox coords
    results_dict = {}
    for class_ in top_n_classes:
        class_scores_dict = class_bbox_scores_dict[class_]
        top_left_corners = [class_scores_dict[meth]['bbox'][0] for meth in methods]
        bottom_right_corners = [class_scores_dict[meth]['bbox'][1] for meth in methods]
        median_top_left = np.median(top_left_corners, 0)
        median_bottom_right = np.median(bottom_right_corners, 0)

        results_dict[class_] = (median_top_left, median_bottom_right)

    return results_dict


def draw(img, results_dict):
    """
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
	"""
    img2 = img.copy()
    # pick a N distinct labels based on number of objects
    LOGGER.DEBUG_IC(results_dict)
    for label, bbox in results_dict.items():
        cv2.rectangle(img2, ic(tuple(bbox[0].astype(int))), tuple(bbox[1].astype(int)), (255, 0, 0), 2)
        cv2.putText(img2, label, (int(bbox[0][0]), int(bbox[1][1]) + 13), 0, 0.5, (255, 0, 0))

    cv2.imshow('test', img2)
    cv2.waitKey(0)


def run(png_path, pyramids):
    test_image = cv2.imread(png_path)

    # Find bbox dims in training image
    test_image_bboxes = get_bbox_dims(test_image)
    LOGGER.DEBUG_IC(test_image_bboxes)
    N = len(test_image_bboxes)

    # Extract image classes from file names
    file_names = os.listdir(cfg.get('TrainingDataPath'))
    class_names = [re.search('-(.*)\.', name).group(1) for name in file_names]

    # filter only needed templates
    templates = {}
    for i, pyramid in enumerate(pyramids):
        class_name = class_names[i]
        templates[class_name] = extract_templates_from_pyramid(pyramid, test_image_bboxes)

    # set background to 0 for extracted templates
    for template_levels_dict in templates.values():
        for template_levels in template_levels_dict.values():
            for template in template_levels:
                _, thresh = cv2.threshold(cv2.cvtColor(np.array(template), cv2.COLOR_BGR2GRAY), 245, 255, cv2.THRESH_BINARY)
                template[thresh == 255] = 0

    # set background to 0 for test image
    transparent_test_image = test_image.copy()
    _, thresh = cv2.threshold(cv2.cvtColor(transparent_test_image, cv2.COLOR_BGR2GRAY), 245, 255, cv2.THRESH_BINARY)
    transparent_test_image[thresh == 255] = 0

    # template match
    final_bboxes_dict = template_match(transparent_test_image, N, templates)

    if cfg.getboolean('ShowResults'):
        draw(test_image, final_bboxes_dict)

    return final_bboxes_dict


def main():
    # Parsing image file and declaring necessary params for pyramid generation
    parser = argparse.ArgumentParser()
    parser.add_argument("png_path", help="Path to a test image")
    args = parser.parse_args()

    training_data_path = cfg.get('TrainingDataPath')
    pyramids = generate_pyramids(training_data_path)

    run(args.png_path, pyramids)


if __name__ == "__main__":
    main()
