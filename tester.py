import sys
import os
import time
import numpy as np
from src.util import Algorithm, config, sort_ascending, parse_annotation_txt_files
from src.task_1 import algorithm as find_angle
from src.task_2 import algorithm as template_match
from src.task_3 import algorithm as sift


class Tester:
    def __init__(self, test_option=Algorithm.ALL):
        self.test_option = test_option
        self.config = None
        self.img_path = ''
        self.annotations_path = ''

    def run(self):
        results = []

        def test1():
            results.append(self.__test_1())

        def test2():
            results.append(self.__test_2_3(Algorithm.TEMPLATE_MATCHING))

        def test3():
            results.append(self.__test_2_3(Algorithm.SIFT))

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
            print(f'[ERROR] OPTION {self.test_option} DOES NOT EXIST.')
            sys.exit(1)

        options[self.test_option]()

        if False in results:
            fail_indices = [i + 1 for i, x in enumerate(results) if not x]
            print(f'[FAILURE] TEST {str(fail_indices)[1:-1]} FAILED')
        else:
            print('[SUCCESS] ALL TESTS PASSED!')

        return

    def __test_1(self) -> bool:
        self.config = config(Algorithm.FIND_ANGLE)
        self.img_path = os.path.join(os.path.dirname(__file__), self.config.get('ImgDataPath'))
        self.annotations_path = os.path.join(os.path.dirname(__file__), self.config.get('ListPath'))

        images = {}
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

            angle = round(float(angle))
            theta = round(float(theta))

            print(name)
            print(f'Expected angle: {angle}°')
            print(f'Found angle: {theta}°')
            if angle == theta:
                print('TEST PASSED')
                passes += 1
            else:
                print('TEST FAILED')
                fails += 1

            indiv_end_time = round(time.time() - indiv_start_time, 3)
            print(f'RUNTIME: {indiv_end_time}s\n')

        end_time = round(time.time() - start_time, 3)
        print(f'PASSES: {passes} | FAILS: {fails} | RUNTIME: {end_time}s')
        return passes == len(images)

    def __test_2_3(self, opt) -> bool:
        if (opt != Algorithm.TEMPLATE_MATCHING) and (opt != Algorithm.SIFT):
            print("[ERROR] INVALID TEST OPTION SPECIFIED FOR TEST 2/3!")
            return False

        self.config = config(Algorithm.SIFT)

        is_sift = (opt == Algorithm.SIFT)

        self.annotations_path = self.config.get('AnnotationsPath')
        label_annotations = parse_annotation_txt_files(self.annotations_path)

        training_data_path = self.config.get('TrainingDataPath')
        templates = sift.read_template_dir(training_data_path) if is_sift else template_match.generate_pyramids(
            training_data_path)

        self.img_path = self.config.get('TestImgDataPath')
        test_img_paths = [os.path.join(self.img_path, f) for f in os.listdir(self.img_path) if f.endswith('.png')]
        test_img_paths = sort_ascending(test_img_paths)

        passes = 0
        fails = 0

        start_time = time.time()
        for i, img_path in enumerate(test_img_paths):
            indiv_start_time = time.time()
            results_dict = sift.run(img_path, templates) if is_sift else template_match.run(img_path, templates)
            labels_dict = label_annotations[i]

            class_labels = set(labels_dict.keys())
            class_segmented = set(results_dict.keys())

            if class_labels == class_segmented:
                for class_ in class_labels:
                    segmented_bbox = results_dict[class_]
                    measured_bbox = labels_dict[class_]
                    # check if segmented bbox within 30 pixels of real one
                    top_left_diff = (segmented_bbox[0] - measured_bbox[0]) ** 2
                    bottom_right_diff = (segmented_bbox[1] - measured_bbox[1]) ** 2
                    total_diff = np.sum((top_left_diff + bottom_right_diff)) ** 0.5

                    if total_diff > 20:
                        print(f'BBOX for class: {class_} not a good match')

                print(f'[SUCCESS] match for img: {img_path}')
                passes += 1
            else:
                print(f'[ERROR] img_f: {img_path} failed, not all classes match')
                fails += 1

            invid_end_time = round(time.time() - indiv_start_time)
            print(f'RUNTIME: {invid_end_time}s\n')

        end_time = round(time.time() - start_time, 3)
        if end_time > 60:
            print(f'RUNTIME: {int(end_time/60)}mins {round(end_time%60, 3)}s')
        else:
            print(f'RUNTIME: {end_time}s')
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
        print(f'[ERROR] "{user_input}" NOT DEFINED!')


if __name__ == '__main__':
    main()
