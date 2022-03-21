import cv2
import os
import matplotlib.pyplot as plt


def get_images(folder_path):
    '''
    Reads all jpgs and pngs in a folder and returns them in a list

    Parameters
    ----------
    folder_path : string
        the relative or absolute path to the folder

    Returns
    -------
    collector : list
        an array containing all images in that folder as matrices
    '''
    valid_types = ['.jpg', '.png']
    collector = [ ]

    for file_name in os.listdir(folder_path):
        _, file_type = os.path.splitext(file_name)
        if file_type.lower() not in valid_types:
            continue

        image = cv2.imread(folder_path + '/' + file_name)
        collector.append(image)

    return collector


def draw_gaussian_pyramid(pyramid, levels, rotations):
    '''
    Utility function for drawing gaussian pyramids

    Parameters
    ----------
    pyramid : dictionary
        contains all pyramids to be drawn

    levels : int
        amount of levels per pyramid

    rotations : int
        amount of rotations per image
    '''
    aspect_params = {'height_ratios': [(2**x) for x in range(0, levels)]}
    fig, axarr = plt.subplots(levels, rotations, gridspec_kw=aspect_params)

    for rot in range(rotations):
        rot_drawn = False
        for lvl in range(levels - 1, -1, -1):
            #key = '' + str(rot) + '_' + str(lvl)
            el = axarr[levels - lvl - 1, rot]
            el.imshow(pyramid[rot][lvl])
            el.axis('off')

            if rot_drawn == False:
                fig.text(1 / rotations * rot, 0.05, f'{360 / rotations * rot}°')
                rot_drawn = True

    plt.tight_layout()
    plt.show()
