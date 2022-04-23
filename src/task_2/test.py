import os
import time
from template_match import algorithm_run
import numpy as np
import re
import configparser as cfgp

config = cfgp.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../settings.INI'))
config = config['TASK2_3']

def sort_ascending(list):
    return sorted(list, key = lambda e: int(e.split('_')[-1].split('.')[0]))

def parse_annotation_txt_files(path_to_txt_files):
    '''
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
    '''
    txt_paths = [f for f in os.listdir(path_to_txt_files) if f.endswith('.txt')]
    txt_paths = sort_ascending(txt_paths)

    labels_dict_list = []
    for f in txt_paths:
        labels_dict = {}
        with open(os.path.join(path_to_txt_files,f),'r') as f:
            lines = [line.rstrip().split(',') for line in f.readlines()]
            for line in lines:
                class_ = line[0]
                top_left =np.asarray(re.findall(r'\d+',','.join(line[1:3]))).astype(int)
                bottom_right = np.asarray(re.findall(r'\d+',','.join(line[3:]))).astype(int)
                labels_dict[class_] = (top_left,bottom_right)

        labels_dict_list.append(labels_dict)

    return labels_dict_list


labels_dir = config.get('AnnotationsPath')
labels_dict_list = parse_annotation_txt_files(labels_dir)

test_data_dir = config.get('TestImgDataPath')
test_img_paths = [os.path.join(test_data_dir,f) for f in os.listdir(test_data_dir) if f.endswith('.png')]
test_img_paths = sort_ascending(test_img_paths)

start_time = time.time()
for idx,img_f in enumerate(test_img_paths):
    indiv_start_time = time.time()
    results_dict = algorithm_run(img_f)
    labels_dict = labels_dict_list[idx]

    class_labels = set(labels_dict.keys())
    class_segmented = set(results_dict.keys())

    if class_labels == class_segmented:
        for class_ in class_labels:
            segmented_bbox = results_dict[class_]
            measured_bbox = labels_dict[class_]
            # check if segmented bbox within 30 pixels of real one
            top_left_diff = (segmented_bbox[0] - measured_bbox[0])**2
            bottom_right_diff = (segmented_bbox[1] - measured_bbox[1])**2
            total_diff = np.sum((top_left_diff + bottom_right_diff))**0.5

            if total_diff>20:
                print(f'BBOX for class: {class_} not a good match')

        print(f'[SUCCESS] match for img: {img_f}')
    else:
        print(f'[ERROR] img_f: {img_f} failed, not all classes match')

    indiv_end_time = round(time.time() - indiv_start_time, 3)
    print(f'RUNTIME: {indiv_end_time}s\n')

end_time = round(time.time() - start_time, 3)
if end_time > 60:
    print(f'RUNTIME: {int(end_time/60)}mins {round(end_time%60, 3)}s')
else:
    print(f'RUNTIME: {end_time}s')
