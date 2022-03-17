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


def template_match(): # todo: more params
    '''
    Slides all scaled and rotated templates over the image, measuring the similarities across
    the X and Y axes. Something something similarity score (research).

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
