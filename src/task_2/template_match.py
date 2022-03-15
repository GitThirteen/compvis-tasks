import cv2
import os
import numpy as np
import configparser as cfgp

config = cfgp.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../settings.INI'))
config = config['TASK2']

def create_gaussian_pyramid(template, rotations): # todo: more params, alternatively: split stuff into own func
    '''
    Creates a gaussian pyramid for every training image via (gaussian) burring and subsequent
    subsampling. Furthermore, for a specified amount of rotations, generates additional
    pyramids and adds them to the dictionary (?).

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

def main():
    # TODO
    return

if __name__ == "__main__":
    main()
