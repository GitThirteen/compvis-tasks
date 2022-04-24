import cv2
import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from ..util import config, Algorithm, Logger

cfg = config(Algorithm.FIND_ANGLE)
LOGGER = Logger.get()

def check_acute_seperation(edge_map, unique_lines):
    '''
    Checks seperation between edges to find whether acute or obtuse seperation.
    Finds direction vector of each edge using the intersecting point and computes the dot product to find the angle between the edges.

    Parameters
    ----------
    acute_angle : float
        acute angle between lines found from the hough transform

    edge_map : np.ndarray
        edge map of input gray scale image

    unique_lines : list
        list containing gradient and y intercept parameter tuples for 2 distinct lines, [(m1,c1), (m2,c2)]

    Returns
    -------
    acute_bool : bool
        Boolean value determined by whether or not seperation found to be acute
    '''

    # get index position array of edge pixels
    pos_array = np.argwhere(edge_map)

    # calculate intersection point
    m1, c1 = unique_lines[0]
    m2, c2 = unique_lines[1]

    x_int = (c2 - c1) / (m1 - m2)
    y_int = m1 * x_int + c1
    intersection_point = np.asarray([y_int, x_int])  # y=i, x=j, flipped for array indexing

    LOGGER.DEBUG(f'intersection point: {intersection_point}')

    # find distance of edge pixels from intersection point
    pos_frm_int_array = pos_array - intersection_point  # find difference in i,j indices
    dist_frm_int_array = np.sum(pos_frm_int_array ** 2, 1) ** 0.5  # find abs difference

    # select furthest pixel to represent edge 1
    edge_point_1_idx = np.argmax(dist_frm_int_array)  # picks an edge end point as first edge
    edge_point_1 = pos_array[edge_point_1_idx]
    thresh = dist_frm_int_array[edge_point_1_idx] / 10

    # calculate distance of edge pixels from edge_point_1
    pos_frm_edge_1_array = pos_array - edge_point_1  # find difference in i,j indices
    dist_frm_edge_1_array = (np.sum(pos_frm_edge_1_array ** 2, 1) ** 0.5)  # find abs difference

    # select pixel furthest away from intersect and also thresh distance away from edge_1
    threshold_array = np.argwhere(np.where(dist_frm_edge_1_array > thresh, 1, 0))
    threshold_pos_array = pos_array[threshold_array.flatten()]

    threshold_pos_frm_int_array = threshold_pos_array - intersection_point
    threshold_dist_frm_int_array = np.sum(threshold_pos_frm_int_array ** 2, 1) ** 0.5
    edge_point_2_idx = np.argmax(threshold_dist_frm_int_array)

    LOGGER.DEBUG(f'edge point 1:{pos_array[edge_point_1_idx]}')
    LOGGER.DEBUG(f'edge point 2:{pos_array[edge_point_2_idx]}')

    # find unit vectors corresponding to direction of edge points
    edge_vec_1 = pos_frm_int_array[edge_point_1_idx]
    edge_vec_2 = threshold_pos_frm_int_array[edge_point_2_idx]

    edge_unit_vec_1 = edge_vec_1 / np.linalg.norm(edge_vec_1)
    edge_unit_vec_2 = edge_vec_2 / np.linalg.norm(edge_vec_2)

    LOGGER.DEBUG(f'edge unit vec 1: {edge_unit_vec_1}, edge unit vec 2: {edge_unit_vec_2}')

    # TODO: Clipping doesn't change result atm, not sure if it's needed after we fix edge points?
    angle = np.arccos(np.clip(np.dot(edge_unit_vec_1, edge_unit_vec_2), -1.0, 1.0))

    LOGGER.DEBUG(angle)
    LOGGER.DEBUG(np.pi / 2)
    LOGGER.DEBUG(f'angle from arcos: {angle * 180 / np.pi}')

    acute_bool = angle <= np.pi / 2
    return acute_bool


def run(png_path):
    '''
    Reads a png with opencv, runs canny edge detection and applies a hough transform to extract lines
    Checks if acute or obtuse seperation and returns the appropriate angle.

    Parameters
    ----------
    png_path : str
        path to png image containing lines

    Returns
    -------
    angle : float
        angle between lines in degrees
    '''

    # read image
    img = cv2.imread(png_path)
    # convert to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find edges
    edges = auto_canny(gray_img, 0.33)

    '''
    Figure out way to select best threshold value / resolution based on thickness of lines / edges
    currently calculating a bbox which contains the edge and setting thresh to the value of the box's diagonal
    '''
    # find edge bbox
    pos_array = np.argwhere(edges)
    i_max, i_min = np.amax(pos_array[:, 0]), np.amin(pos_array[:, 0])
    j_max, j_min = np.amax(pos_array[:, 1]), np.amin(pos_array[:, 1])
    bbox_diag = ((i_max - i_min) ** 2 + (j_max - j_min) ** 2) ** 0.5

    # Loop over different thresholds and stop when 2 distinct lines found
    for val in np.linspace(1, 3, 10):
        thresh = int(round(bbox_diag / val))

        # apply a hough line transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, thresh)

        if (lines is None) or (len(lines) < 2):
            continue

        lines = lines.squeeze()
        # remove parallel lines - rewrite in fewer lines?
        theta_list = []
        unique_lines = []
        for line in lines:
            theta = line[1]
            if theta not in theta_list:
                theta_list.append(theta)
                unique_lines.append(line)

        unique_lines = np.asarray(unique_lines)
        if len(unique_lines) == 2:
            break

    if cfg.getboolean('ShowHoughLines'):
        draw_houghlines(lines, img)

    # check for vertical lines:
    if len(np.nonzero(unique_lines[:, 1])[0]) < len(unique_lines):
        # rotate edges by 90 to turn vertical line to horizontal
        LOGGER.INFO('Vertical lines in image, rotating to avoid inf gradients')

        h, w = edges.shape
        offset = w
        edges = np.rot90(edges)
        vert_bool = True
    else:
        vert_bool = False

    # array to store line param tuples (m,c)
    unique_cartesian_lines = []

    # loop over lines found via the hough line transform
    for line in unique_lines:
        # extract rho theta values for each line
        rho, theta = line
        if vert_bool: theta -= np.pi / 2
        LOGGER.DEBUG(f'line polar params (rho,theta): {rho}, {theta}')

        # calc line cartesian params
        cosx = np.cos(theta)
        sinx = np.sin(theta)
        m = -1 / np.tan(theta)
        x0 = cosx * rho
        y0 = sinx * rho
        c = y0 - m * x0

        if vert_bool: c += offset
        LOGGER.DEBUG(f'line cartesian params (m,c): {m}, {c}')

        # append param tuple to list
        unique_cartesian_lines.append((m, c))

    ''' at this step len angles should = 2, if more than 2 angles at this stage need another filtering step'''
    if len(unique_lines) == 2:
        # calculate angle of incidences to find angle between lines
        m1, c1 = unique_cartesian_lines[0]
        m2, c2 = unique_cartesian_lines[1]

        # show lines and edges
        if cfg.getboolean('ShowHoughLines'):
            x = np.arange(0, edges.shape[1], 1)
            plt.imshow(edges)
            plt.scatter(x, m1 * x + c1, s=0.5)
            plt.scatter(x, m2 * x + c2, s=0.5)
            plt.show()

        angle_1 = np.arctan(m1)
        angle_2 = np.arctan(m2)

        # find absolute acute angle
        angle = np.abs(angle_1 - angle_2)
        acute_angle = np.pi - angle if angle > np.pi / 2 else angle

        # check if acute seperation true
        acute_bool = check_acute_seperation(edges, unique_cartesian_lines)

        # return angle in degrees
        if acute_bool:
            return 180 / np.pi * acute_angle
        else:
            return 180 / np.pi * (np.pi - acute_angle)

    else:
        LOGGER.ERROR('MORE THAN 2 LINES FOUND. ABORTING.')
        return sys.exit(1)


def draw_houghlines(lines, img):
    for line in lines:
        rho, theta = line

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        plt.imshow(img)
        plt.show()


def auto_canny(img, sigma):
    # Calculate median of pixel intensities
    pixel_median = np.median(img)

    # Use median to find upper and lower bound for canny edge detection
    lower = int(max(0, (1.0 - sigma) * pixel_median))
    upper = int(min(255, (1.0 + sigma) * pixel_median))

    # Perform edge detection with computed upper and lower bounds
    edges = cv2.Canny(img, lower, upper, L2gradient=True)
    return edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("png_path", help="Path to a png image containing lines")
    args = parser.parse_args()

    theta = run(args.png_path)
    LOGGER.DEBUG(f'angle found:{theta:.3f}')


if __name__ == "__main__":
    main()
