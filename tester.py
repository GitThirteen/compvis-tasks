import sys
import os
import time
import csv
import cv2
import numpy as np
from src.util import Algorithm, Logger, config, sort_ascending, parse_annotation_txt_files, get_bbox_iou
from src.task_1 import algorithm as find_angle
from src.task_2 import algorithm as template_match
from src.task_3 import algorithm as sift

LOGGER = Logger.get()

class Tester:
    def __init__(self, test_option=Algorithm.ALL):
        self.test_option = test_option
        self.config = None
        self.img_path = ''
        self.annotations_path = ''

    def run(self):
        results = [True, True, True]

        def test1():
            results[0] = self.__test_1()

        def test2():
            results[1] = self.__test_2_3(Algorithm.TEMPLATE_MATCHING)

        def test3():
            results[2] = self.__test_2_3(Algorithm.SIFT)

        def all():
            test1()
            test2()
            test3()

        options = {
            Algorithm.FIND_ANGLE: test1,
            Algorithm.TEMPLATE_MATCHING: test2,
            Algorithm.SIFT: test3,
            Algorithm.ALL: all
        }

        if self.test_option not in options.keys():
            # In case an invalid test_option was specified
            LOGGER.ERROR(f'OPTION {self.test_option} DOES NOT EXIST.')
            sys.exit(1)

        options[self.test_option]()

        if False in results:
            fail_indices = [i + 1 for i, x in enumerate(results) if not x]
            LOGGER.ERROR(f'TEST {str(fail_indices)[1:-1]} FAILED')
        else:
            LOGGER.SUCCESS(f'ALL TESTS PASSED!')

        return

    def __test_1(self) -> bool:
        self.config = config(Algorithm.FIND_ANGLE)
        self.img_path = os.path.join(os.path.dirname(__file__), self.config.get('ImgDataPath'))
        self.annotations_path = os.path.join(os.path.dirname(__file__), self.config.get('ListPath'))

        images = { }
        with open(self.annotations_path) as f:
            lines = f.readlines()
            for line in lines:
                frags = line.split(',')
                images[frags[0]] = frags[1]

        passes = 0
        fails = 0

        start_time = time.time()
        for name, angle in images.items():
            indiv_start_time = time.time()
            path = self.img_path + '/' + name
            theta = find_angle.run(path)

            angle = float(angle)
            theta = round(theta, 2)

            LOGGER.INFO(name)
            LOGGER.INFO(f'Expected angle: {angle}°')
            LOGGER.INFO(f'Found angle: {theta}°')
            if angle == theta:
                LOGGER.SUCCESS('TEST PASSED')
                passes += 1
            else:
                LOGGER.ERROR('TEST FAILED')
                fails += 1

            indiv_end_time = round(time.time() - indiv_start_time, 3)
            LOGGER.INFO(f'RUNTIME: {indiv_end_time}s\n')

        end_time = round(time.time() - start_time, 3)
        LOGGER.INFO(f'PASSES: {passes} | FAILS: {fails} | RUNTIME: {end_time}s')
        return passes == len(images)

    def __test_2_3(self, opt) -> bool:
        if (opt != Algorithm.TEMPLATE_MATCHING) and (opt != Algorithm.SIFT):
            LOGGER.ERROR('INVALID TEST OPTION SPECIFIED FOR TEST 2/3!')
            return False

        self.config = config(Algorithm.SIFT)
        is_sift = (opt == Algorithm.SIFT)

        self.annotations_path = self.config.get('AnnotationsPath')
        label_annotations = parse_annotation_txt_files(self.annotations_path)

        training_data_path = self.config.get('TrainingDataPath')
        if is_sift:
            templates = sift.read_template_dir(training_data_path)
            templates = sift.get_template_kp_des(templates)
        else:
            templates = template_match.generate_pyramids(training_data_path)

        self.img_path = self.config.get('TestImgDataPath')
        test_img_paths = [os.path.join(self.img_path, f) for f in os.listdir(self.img_path) if f.endswith('.png')]
        test_img_paths = sort_ascending(test_img_paths)

        passes = 0
        fails = 0

        if self.config.getboolean('PreciseData'):
            num_templates = len(templates)
            header = ['Image', 'Runtime', 'OUTCOME', 'Overlap', 'False Positive Rate', 'True Positive Rate', 'Precision', 'Recall']
            data = []

        start_time = time.time()
        for i, img_path in enumerate(test_img_paths):
            indiv_start_time = time.time()
            results_dict = sift.run(img_path, templates) if is_sift else template_match.run(img_path, templates)
            labels_dict = label_annotations[i]

            class_labels = set(labels_dict.keys())
            class_segmented = set(results_dict.keys())

            overlaps = [ ]

            if self.config.getboolean('PreciseData'):
                negatives = num_templates - len(class_labels)
                true_positives = len([class_ for class_ in class_segmented if class_ in class_labels])
                false_positives = len([class_ for class_ in class_segmented if class_ not in class_labels])
                false_negatives = len([class_ for class_ in class_labels if class_ not in class_segmented])

                fpr = (false_positives / negatives) * 100
                tpr = (true_positives / len(class_labels)) * 100
                precision = (true_positives / (true_positives + false_positives)) * 100
                recall = (true_positives / (true_positives + false_negatives)) * 100

                #accuracy = ((true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives))*100
            
            #if class_labels == class_segmented:

            for label in class_labels:
                if (label not in results_dict) or (label not in labels_dict):
                    # assuming 0 overlap if wrongly labeled or label not found
                    overlaps.append(0)
                    continue

                segmented_bbox = results_dict[label]
                measured_bbox = labels_dict[label]

                overlap = get_bbox_iou(segmented_bbox, measured_bbox) * 100
                overlaps.append(overlap)

                # check if segmented bbox within 30 pixels of real one
                top_left_diff = (segmented_bbox[0] - measured_bbox[0]) ** 2
                bottom_right_diff = (segmented_bbox[1] - measured_bbox[1]) ** 2
                total_diff = np.sum((top_left_diff + bottom_right_diff)) ** 0.5

                if total_diff > 10:
                    LOGGER.WARNING(f'BBOX for class: {label} not a good match')

            if class_labels == class_segmented:
                outcome = 'SUCCESS'
                LOGGER.SUCCESS(f'match for img: {img_path}')
                passes += 1
            else:
                outcome = 'FAIL'
                LOGGER.ERROR(f'img_f: {img_path} failed, not all classes match')
                fails += 1

            invid_end_time = round(time.time() - indiv_start_time, 3)
            LOGGER.INFO(f'RUNTIME: {invid_end_time}s\n')

            if self.config.getboolean('PreciseData'):
                overlap = round(np.sum(overlaps) / len(overlaps), 2)
                LOGGER.INFO(f'Overlap: {overlap}% | TPR: {tpr}% | FPR: {fpr}% | Precision: {precision}% | Recall: {recall}%\n')
                data.append([f'Image {i+1}', f'{invid_end_time}s', outcome, f'{overlap}%', f'{fpr}%', f'{tpr}%', f'{precision}%', f'{recall}%'])

        end_time = round(time.time() - start_time, 3)
        if end_time > 60:
            final_runtime = f'{int(end_time/60)}m {round(end_time%60, 3)}s'
        else:
            final_runtime = f'{end_time}s'

        LOGGER.INFO(f'PASSES: {passes} | FAILS: {fails} | RUNTIME: {final_runtime}')

        if self.config.getboolean('PreciseData'):
            data.append([f'TOTAL RUNTIME: {final_runtime}'])

            with open('run_data.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Task 3' if opt == Algorithm.SIFT else 'Task 2'])

                # write the header
                writer.writerow(header)
                # write multiple rows
                writer.writerows(data)

        return passes == len(test_img_paths)


def main():
    user_input = input('Which algorithm would you like to test?\n(Options: 1, 2, 3, all)\n')

    options = {
        '1': Algorithm.FIND_ANGLE,
        '2': Algorithm.TEMPLATE_MATCHING,
        '3': Algorithm.SIFT,
        'all': Algorithm.ALL
    }

    if user_input in options.keys():
        test_option = options[user_input.lower()]
        Tester(test_option).run()
    else:
        LOGGER.ERROR(f'"{user_input}" NOT DEFINED!')


if __name__ == '__main__':
    main()
