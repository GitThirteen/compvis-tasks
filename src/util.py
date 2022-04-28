import os
import cv2
import re
import time
from enum import Enum
from icecream import ic
import configparser as cfgp
import matplotlib.pyplot as plt
import numpy as np


class Algorithm(Enum):
    FIND_ANGLE = 1
    TEMPLATE_MATCHING = 2
    SIFT = 3
    ALL = 4


class bcolors:
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class Logger:
    __instance = None

    def __init__(self, cfg):
        self.cfg = cfg

    def ERROR(self, content):
        if self.cfg.getboolean('ShowError'):
            print(f'{bcolors.FAIL}{bcolors.BOLD}[ERROR]{bcolors.ENDC} {content}')

    def WARNING(self, content):
        if self.cfg.getboolean('ShowWarning'):
            print(f'{bcolors.WARNING}{bcolors.BOLD}[WARNING]{bcolors.ENDC} {content}')

    def INFO(self, content):
        if self.cfg.getboolean('ShowInfo'):
            print(f'{bcolors.OKCYAN}{bcolors.BOLD}[INFO]{bcolors.ENDC} {content}')

    def SUCCESS(self, content):
        if self.cfg.getboolean('ShowSuccess'):
            print(f'{bcolors.OKGREEN}{bcolors.BOLD}[SUCCESS]{bcolors.ENDC} {content}')

    def DEBUG(self, content):
        if self.cfg.getboolean('DebugMode'):
            print(f'{bcolors.BOLD}[DEBUG]{bcolors.ENDC} {content}')

    def DEBUG_IC(self, *args):
        if self.cfg.getboolean('DebugMode'):
            ic(args)

    def start_timer(self):
        if self.cfg.getboolean('DebugMode'):
            self.time = time.time()

    def measure_time_diff(self, name=''):
        if self.cfg.getboolean('DebugMode'):
            print(f"[TIMER: {name}] " + str(round(time.time() - self.time, 3)))
            self.time = time.time()

    @staticmethod
    def get() -> 'Logger':
        if Logger.__instance == None:
            cfg = cfgp.ConfigParser()
            cfg.read(os.path.join(os.path.dirname(__file__), 'settings.INI'))
            cfg = cfg['LOGGER_OPTIONS']
            Logger.__instance = Logger(cfg)

        return Logger.__instance


def config(test_option) -> cfgp.SectionProxy:
    cfg = cfgp.ConfigParser()
    cfg.read(os.path.join(os.path.dirname(__file__), 'settings.INI'))

    if test_option == Algorithm.FIND_ANGLE:
        return cfg['TASK1']
    else:
        return cfg['TASK2_3']


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

def get_bbox_dims(img, sift=False):
    """
	Finds dimensions for bounding boxes in passed in image.

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
    dims_list = []
    new = img.copy()

    # Blur the image
    blur = cv2.GaussianBlur(new, (11, 11), 3)
    grey = blur if sift else cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
 
    # Convert the grayscale image to binary
    _, binary = cv2.threshold(grey, 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(~binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w * h) > 500:
            dims = (w, h)
            cv2.rectangle(new, (x, y), (x + w, y + h), (255, 0, 0), 5)
            dims_dict[i] = (np.asarray([x, y]), np.asarray([x + w, y + h]))
            dims_list.append(dims)
            i += 1

    # cv2.imshow('bboxs detected from raw image',new)
    # cv2.waitKey(0)

    # cv2.imshow('bboxs detected from raw image',new)
    # cv2.waitKey(0)
    if sift:
        return dims_dict

    return dims_list


def get_bbox_iou(bbox1, bbox2):
    if (bbox1[0] == bbox2[0]).all() and (bbox1[1] == bbox2[1]).all():
        return 1.0

    # intersection area (inner pixels)
    x_left = max(bbox1[0][0], bbox2[0][0])
    y_top = max(bbox1[0][1], bbox2[0][1])
    x_right = min(bbox1[1][0], bbox2[1][0])
    y_bottom = min(bbox1[1][1], bbox2[1][1])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # area for each bbox
    bbox1_area = (bbox1[1][0] - bbox1[0][0] + 1) * (bbox1[1][1] - bbox1[0][1] + 1)
    bbox2_area = (bbox2[1][0] - bbox2[0][0] + 1) * (bbox2[1][1] - bbox2[0][1] + 1)

    # inclusion-exclusion principle
    union_area = float(bbox1_area + bbox2_area - intersection_area)

    return intersection_area / union_area
