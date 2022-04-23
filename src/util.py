import os
import cv2
import re
from enum import Enum
import configparser as cfgp
import matplotlib.pyplot as plt
import numpy as np


class Algorithm(Enum):
    FIND_ANGLE = 1
    TEMPLATE_MATCHING = 2
    SIFT = 3
    ALL = 4


def config(test_option) -> cfgp.SectionProxy:
    config = cfgp.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'settings.INI'))

    if test_option == Algorithm.FIND_ANGLE:
        return config['TASK1']
    else:
        return config['TASK2_3']


def sort_ascending(list) -> list:
    return sorted(list, key=lambda e: int(e.split('_')[-1].split('.')[0]))


def get_images(folder_path) -> list:
    """
    Reads all jpgs and pngs in a folder and returns them in a list

    Parameters
    ----------
    folder_path : string
        the relative or absolute path to the folder

    Returns
    -------
    collector : list
        an array containing all images in that folder as matrices
    """
    valid_types = ['.jpg', '.png']
    collector = []

    for file_name in os.listdir(folder_path):
        _, file_type = os.path.splitext(file_name)
        if file_type.lower() not in valid_types:
            continue

        image = cv2.imread(folder_path + '/' + file_name)
        collector.append(image)

    return collector


def parse_annotation_txt_files(path_to_txt_files) -> list:
    """
    Parses directory of annotation text files with format:
    class, bbox top left, bbox bottom right

    Parameters
    ----------
    path_to_txt_files : str
    path to directory containing annotation text files

    Returns
    -------
    labels_dict_list : list
    list of dictionaries with k,v pairs of class labels : bbox corners
    """
    txt_paths = [f for f in os.listdir(path_to_txt_files) if f.endswith('.txt')]
    txt_paths = sort_ascending(txt_paths)

    labels_dict_list = []
    for f in txt_paths:
        labels_dict = {}
        with open(os.path.join(path_to_txt_files, f), 'r') as f:
            lines = [line.rstrip().split(',') for line in f.readlines()]
            for line in lines:
                class_ = line[0]
                top_left = np.asarray(re.findall(r'\d+', ','.join(line[1:3]))).astype(int)
                bottom_right = np.asarray(re.findall(r'\d+', ','.join(line[3:]))).astype(int)
                labels_dict[class_] = (top_left, bottom_right)

        labels_dict_list.append(labels_dict)

    return labels_dict_list


def draw_gaussian_pyramid(pyramid, levels, rotations) -> None:
    """
    Utility function for drawing gaussian pyramids

    Parameters
    ----------
    pyramid : dictionary
        contains all pyramids to be drawn

    levels : int
        amount of levels per pyramid

    rotations : int
        amount of rotations per image
    """
    aspect_params = {'height_ratios': [(2 ** x) for x in range(0, levels)]}
    fig, axarr = plt.subplots(levels, rotations, gridspec_kw=aspect_params)

    for rot in range(rotations):
        rot_drawn = False
        for lvl in range(levels - 1, -1, -1):
            # key = '' + str(rot) + '_' + str(lvl)
            el = axarr[levels - lvl - 1, rot]
            el.imshow(pyramid[rot][lvl])
            el.axis('off')

            if rot_drawn == False:
                fig.text(1 / rotations * rot, 0.05, f'{360 / rotations * rot}°')
                rot_drawn = True

    plt.tight_layout()
    plt.show()