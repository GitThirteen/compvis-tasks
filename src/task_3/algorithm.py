import os
import re
import cv2
import sys
import numpy as np
from icecream import ic
from skimage.measure import ransac
from skimage.transform import AffineTransform

from ..util import config, Algorithm, Logger, get_bbox_dims

cfg = config(Algorithm.SIFT)
LOGGER = Logger.get()

def find_bbox_idx(src_pts, obj_bboxs_dict):
    """
    EXPERIMENTAL
    takes a list of points and bins them according to regions defined by obj_bboxs
    returns idx of region with most points.

    Parameters:
    ----------
    src_pts : np.ndarray
        array of shape N x 2 where cols refer to i,j position of src pt on image and rows refer to individual src pts

    obj_bboxs: dict
        dictionary of bboxs of objects defined in test image, key = bbox idx, label = (np.ndarray(topleft),np.ndarray(bottomright))

    Returns:
    --------
    main_bbox_idx: float
        idx of bbox most src pts fall within
    """
    bbox_bins = np.zeros(len(list(obj_bboxs_dict.keys())))

    LOGGER.DEBUG_IC(obj_bboxs_dict)

    for pt in src_pts:
        for idx, bbox in obj_bboxs_dict.items():
            x, y = pt[0], pt[1]

            if (bbox[0][0] <= x <= bbox[1][0]) and (bbox[0][1] <= y <= bbox[1][1]):
                bbox_bins[idx] += 1

                LOGGER.DEBUG_IC(idx, x, y, bbox)

    main_bbox_idx = np.argmax(bbox_bins)
    num_matched_pts = bbox_bins[main_bbox_idx]
    return main_bbox_idx, num_matched_pts


def read_template_dir(training_data_path):
    """
    EXPERIMENTAL
    Reads pngs at training data path using open cv, imgs are converted to grayscale

    Parameters:
    -----------
    training_data_path: str
        Path to to directory containing template pngs

    Returns:
    --------
    templates: dict
        dict of loaded templates: keys = class, values = templates read via cv2.imread
    """
    # load templates
    valid_types = ['.jpg', '.png']
    template_fname_list = [f for f in os.listdir(training_data_path) if os.path.splitext(f)[1] in valid_types]
    templates = [cv2.imread(os.path.join(training_data_path, f), cv2.IMREAD_GRAYSCALE) for f in template_fname_list]

    # extract class from fname
    class_names = [re.search('-(.*)\.', name).group(1) for name in template_fname_list]

    # compose templates_dict
    templates_dict = {class_names[idx]: templates[idx] for idx in range(len(class_names))}
    return templates_dict


def get_template_kp_des(templates_dict):
    """
    Stores key points and descriptors for all templates in a dictionary

    Parameters
    ----------
    templates_dict : dict
        dict of loaded templates: keys = class, values = templates read via cv2.imread

    Returns
    -------
    templates_dict_kp_des: dict
        dict of all classes with template image, key points and descriptors.
        e.g. {class_: {'image': _, 'key_points': _, 'descriptors': _}, ...}
    """
    sift = cv2.SIFT_create(nOctaveLayers=10) # default = 3
    templates_dict_kp_des = { }

    for class_, template in templates_dict.items():
        # Store template key points and descriptors
        kp, des = sift.detectAndCompute(template, None)
        kp_des = {'image': template, 'key_points': kp, 'descriptors': des}
        templates_dict_kp_des[class_] = kp_des

    return templates_dict_kp_des


def sift_matching(img1, template, template_kp, template_des):
    # Input : image1, template in opencv format, and template key points and descriptors
    # Output : corresponding keypoints for source and target images
    # Output Format : Numpy matrix of shape: [No. of Correspondences X 2]

    LOGGER.start_timer()

    sift = cv2.SIFT_create(nOctaveLayers=10)
    # surf = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)

    LOGGER.measure_time_diff('SIFTING')

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, template_des, k=2)

    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2,k=2)

    LOGGER.measure_time_diff('TREE')

    # Lowe's Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst_pts = np.float32([template_kp[m.trainIdx].pt for m in good]).reshape(-1, 2)

    LOGGER.DEBUG_IC(len(src_pts), len(dst_pts))
    LOGGER.measure_time_diff('RATIO TEST + SRC/DST')
    # Ransac
    try:
        _, inliers = ransac((src_pts, dst_pts), AffineTransform, min_samples=4, residual_threshold=8, max_trials=100)
        LOGGER.measure_time_diff('RANSAC')
    except ValueError:
        LOGGER.DEBUG('No matches')
        return 1
    if inliers is not None:  # inliers found
        n_inliers = np.sum(inliers)

        inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
        inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
        placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
        #image3 = cv2.drawMatches(img1, inlier_keypoints_left, template, inlier_keypoints_right, placeholder_matches,
        #                         None)
        # UNCOMMENT TO SHOW MATCHED FEATURES
        # ----------------------------------
        # cv2.imshow('Matches', image3)
        # cv2.waitKey(0)

        src_pts = np.float32([inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches]).reshape(-1, 2)
        dst_pts = np.float32([inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches]).reshape(-1, 2)

        LOGGER.measure_time_diff('POST-RANSAC')

        return src_pts, dst_pts
    else:
        LOGGER.DEBUG('No inliers')
        return 1


def draw(img, results_dict):
    '''
	Draws a bounding box around each detected object, including its class label.

	Parameters
	----------
	img : np.ndarray
					return value of cv2.imread(img_path), a 3d numpy array

	bboxes : dictionary
					dictionary with the image label as the key and the bbox top left and bottom right stored as tuples in a list as the value.
					e.g. {"hydrant": [top_left, bottom_right], ...}

	Outputs
	-------
	final_image : np.ndarray
					Final image with bboxes and labels represented as a 3d numpy array
	'''
    img2 = img.copy()
    # pick a N distinct labels based on number of objects
    LOGGER.DEBUG_IC(results_dict)
    for label, bbox in results_dict.items():
        cv2.rectangle(img2, ic(tuple(bbox[0].astype(int))), tuple(bbox[1].astype(int)), (255, 0, 0), 2)
        cv2.putText(img2, label, (int(bbox[0][0]), int(bbox[1][1]) + 13), 0, 0.5, (255, 0, 0))

    cv2.imshow('test', img2)
    cv2.waitKey(0)


def run(img_path, templates_dict_kp_des):
    # load img
    color_img = cv2.imread(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # find unclassified bboxs in img
    # key = 1,2,3,...,N value = [np.ndarray(x,y),np.ndarray(x+w,y+h)]
    obj_bboxs_dict = get_bbox_dims(img, True)
    LOGGER.DEBUG_IC(obj_bboxs_dict)

    # initialise empty dict to track pts matched with each class
    obj_class_dict = {idx: {} for idx in list(obj_bboxs_dict.keys())}

    # loop over templates
    for class_, template_data in templates_dict_kp_des.items():
        LOGGER.DEBUG_IC(class_)

        # try matching sift features
        try:
            template, kp, des = template_data['image'], template_data['key_points'], template_data['descriptors']
            src_pts, dst_pts = sift_matching(img, template, kp, des)

            # find main class of object defined by src pts
            bbox_idx, num_matched_pts = find_bbox_idx(src_pts, obj_bboxs_dict)

            LOGGER.DEBUG_IC(bbox_idx, num_matched_pts)

            # update class_bbox_dict
            if 'num_matches' in obj_class_dict[bbox_idx]:
                # only update if num matches has increases
                if num_matched_pts > obj_class_dict[bbox_idx]['num_matches']:
                    obj_class_dict[bbox_idx]['num_matches'] = num_matched_pts
                    obj_class_dict[bbox_idx]['class_'] = class_
            else:
                obj_class_dict[bbox_idx]['num_matches'] = num_matched_pts
                obj_class_dict[bbox_idx]['class_'] = class_

            LOGGER.DEBUG_IC(obj_class_dict)
        except TypeError:
            # no match for template
            pass

        # plot_bboxs(img,class_bbox_dict)

    results = { }
    for bbox_idx, class_data in obj_class_dict.items():
        try:
            class_ = class_data['class_']
            results[class_] = obj_bboxs_dict[bbox_idx]
        except KeyError:
            continue

    if cfg.getboolean('ShowResults'):
        ic(results)
        draw(color_img, results)

    return results


def main():
    img_path = sys.argv[1]

    training_data_path = cfg.get('TrainingDataPath')
    templates_dict = read_template_dir(training_data_path)
    templates_dict_kp_des = get_template_kp_des(templates_dict)

    run(img_path, templates_dict_kp_des)


if __name__ == "__main__":
    main()
