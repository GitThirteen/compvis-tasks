import cv2
import numpy as np 
import sys
import matplotlib.pyplot as plt
from icecream import ic
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import os
import configparser as cfgp

config = cfgp.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../settings.INI'))
config = config['TASK2_3']

def find_bbox_idx(src_pts,obj_bboxs_dict):
    """
    EXPERIMENTAL
    takes a list of points and bins them according to regions defined by obj_bboxs
    returns idx of region with most points.

    Parameters:
    ----------
    src_pts : np.ndarray
        array of shape N x 2 where cols refer to i,j position of src pt on image and rows refer to individual src pts
    
    obj_bboxs: dict
        dictionary of bboxs of objects defined in test image, key = bbox idx, label = (np.ndarray(bottomleft),np.ndarray(topright))
    
    Returns:
    --------
    main_bbox_idx: float
        idx of bbox most src pts fall within
    """
    bbox_bins = np.zeros(len(list(obj_bboxs_dict.keys())))
    ic(obj_bboxs_dict)
    for pt in src_pts:
        for idx,bbox in obj_bboxs_dict.items():
            x,y = pt[0], pt[1]
            if (bbox[0][0]<=x<=bbox[1][0] and bbox[0][1]<=y<=bbox[1][1]):
                ic(idx,x,y,bbox)
                bbox_bins[idx]+=1
    main_bbox_idx = np.argmax(bbox_bins)
    num_matched_pts = bbox_bins[main_bbox_idx]
    return main_bbox_idx,num_matched_pts

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
    template_fname_list = [f for f in os.listdir(training_data_path) if f.endswith('.png')]
    templates = [cv2.imread(os.path.join(training_data_path,f),cv2.IMREAD_GRAYSCALE) for f in template_fname_list]
    
    # extract class from fname
    """ DOESNT WORK for some template f names like gas-station [MICHAEL] """
    classes = [f.split('.')[-2].split('-')[1] for f in template_fname_list]
    
    # compose templates_dict
    templates_dict = {classes[idx]:templates[idx] for idx in range(len(classes))}
    return templates_dict

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
    dims_dict = {}
    new = img.copy()

    # apply an otsu thresh
    ret,binary = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # apply morh closing to close any disconnected parts
    kernel = np.ones((15,15),np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(~binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    idx = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ic(x,y,w,h, w)
        if 1000 < w * h < 5000: # ignore the global image contour and any tiny contours 
            dims = (w, h)
            cv2.rectangle(new,(x,y), (x+w,y+h), (255,0,0), 5)
            dims_dict[idx] = (np.asarray([x,y]),np.asarray([x+w,y+h]))
            idx+=1

    # uncomment to show detected bboxs 
    # ---------------------------------------------
    # cv2.imshow('bboxs detected from raw image',binary)
    # cv2.waitKey(0)
    return dims_dict

def siftMatching(img1, img2):
    # Input : image1 and image2 in opencv format
    # Output : corresponding keypoints for source and target images
    # Output Format : Numpy matrix of shape: [No. of Correspondences X 2] 

    sift = cv2.SIFT_create()
    # surf = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Lowe's Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)
    ic(len(src_pts),len(dst_pts))
    # Ransac
    try:
        model, inliers = ransac((src_pts, dst_pts),AffineTransform, min_samples=4,residual_threshold=8, max_trials=10000)
    except ValueError:
        print('[ERROR], no matches')
        return 1
    if inliers is not None: # inliers found
        n_inliers = np.sum(inliers)

        inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
        inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
        placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
        image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)
        # UNCOMMENT TO SHOW MATCHED FEATURES 
        # ----------------------------------
        cv2.imshow('Matches', image3)
        cv2.waitKey(0)

        src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
        dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

        return src_pts, dst_pts

    else: 
        print('[ERROR], no inliers')
        return 1

if __name__ == "__main__":
    """
    Change this temp set up to a proper one
    CODE SHOULD WORK ASSUMING THE MOST MATCHED FEATURES OCCUR WHEN THE TEMPLATE CLASS = OBJECT CLASS
    """
    img_path = sys.argv[1]
    training_data_path = config.get('TrainingDataPath')
    templates_dict = read_template_dir(training_data_path)
    kernel = np.ones((5,5),np.uint8) # for morph opening

    # load img
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    _, img_thresh = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)
    img[img_thresh == 255] = 0
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # find unclassified bboxs in img
    # key = 1,2,3,...,N value = [np.ndarray(x,y),np.ndarray(x+w,y+h)]
    obj_bboxs_dict = get_bbox_dims(img)
    ic(obj_bboxs_dict)

    # initialise empty dict to track pts matched with each class
    obj_class_dict = {idx:{} for idx in list(obj_bboxs_dict.keys())}
    # loop over templates
    for class_,template in templates_dict.items():
        ic(class_)
        # load template and set background to 0
        _, template_thresh = cv2.threshold(template, 245, 255, cv2.THRESH_BINARY)
        template[template_thresh == 255] = 0
        template = cv2.morphologyEx(template, cv2.MORPH_OPEN, kernel)
        
        # try matching sift features
        try:
            src_pts, dst_pts =  siftMatching(img,template)
            # find main class of object defined by src pts
            bbox_idx, num_matched_pts = find_bbox_idx(src_pts,obj_bboxs_dict)
            ic(bbox_idx,num_matched_pts)
            # update class_bbox_dict
            if 'num_matches' in obj_class_dict[bbox_idx]:
                # only update if num matches has increases
                if num_matched_pts > obj_class_dict[bbox_idx]['num_matches']:
                    obj_class_dict[bbox_idx]['num_matches'] = num_matched_pts
                    obj_class_dict[bbox_idx]['class_'] = class_
            else:
                obj_class_dict[bbox_idx]['num_matches'] = num_matched_pts
                obj_class_dict[bbox_idx]['class_'] = class_
            ic(obj_class_dict)

        except TypeError:
            # no match for template
            pass
        
        # plot_bboxs(img,class_bbox_dict)
        
     
    ic(obj_class_dict)
    fig,ax = plt.subplots(ncols=2)
    ax[0].imshow(img)
    ax[0].scatter(src_pts[:,0],src_pts[:,1])

    ax[1].imshow(template)
    ax[1].scatter(dst_pts[:,0],dst_pts[:,1])

    plt.show()
    # # Initiate SIFT detector
    # sift = cv2.SIFT_create()

    # # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img,None)
    # kp2, des2 = sift.detectAndCompute(template,None)
    
    # # BFMatcher with default params
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2,k=2)
    # # Apply ratio test
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good.append([m])
    
    # # cv.drawMatchesKnn expects list of lists as matches.
    # img3 = cv2.drawMatchesKnn(img,kp1,template,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3)
    # plt.show()
        
    # img=cv2.drawKeypoints(img,kp,img,flags=cv2.     )
    # cv2.imwrite('sift_keypoints.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    

    
    # cv.drawMatchesKnn expects list of lists as matches.
    # img2 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3),plt.show()