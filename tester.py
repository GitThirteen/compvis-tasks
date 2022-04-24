import sys
import os
import time
import csv
import numpy as np
from src.util import Algorithm, Logger, config, sort_ascending, parse_annotation_txt_files
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
            # UNCOMMENT FOR MORE PRECISE DATA
            # num_templates = len(templates)
            templates = sift.get_template_kp_des(templates)
        else:
            templates = template_match.generate_pyramids(training_data_path)
            # UNCOMMENT FOR MORE PRECISE DATA
            # num_templates = len(templates)

        self.img_path = self.config.get('TestImgDataPath')
        test_img_paths = [os.path.join(self.img_path, f) for f in os.listdir(self.img_path) if f.endswith('.png')]
        test_img_paths = sort_ascending(test_img_paths)

        passes = 0
        fails = 0

        # UNCOMMENT FOR MORE PRECISE DATA
        # header = ['Image', 'Runtime', 'SUCCESS/FAIL', 'False Positive Rate', 'True Positive Rate', 'Accuracy']
        # data = []
        start_time = time.time()
        for i, img_path in enumerate(test_img_paths):
            # UNCOMMENT FOR MORE PRECISE DATA
            # success = "FAIL"
            indiv_start_time = time.time()
            results_dict = sift.run(img_path, templates) if is_sift else template_match.run(img_path, templates)
            labels_dict = label_annotations[i]

            class_labels = set(labels_dict.keys())
            class_segmented = set(results_dict.keys())

            # UNCOMMENT FOR MORE PRECISE DATA
            # negatives = num_templates-len(class_labels)
            # true_positives = len([class_ for class_ in class_segmented if class_ in class_labels])
            # false_positives = len([class_ for class_ in class_segmented if class_ not in class_labels])
            # false_negatives = len([class_ for class_ in class_labels if class_ not in class_segmented])
            # true_negatives = negatives - false_negatives - false_positives
            #
            # false_positive_rate = false_positives/negatives
            # true_positive_rate = true_positives/len(class_labels)
            # accuracy = ((true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives))*100

            if class_labels == class_segmented:
                for class_ in class_labels:
                    segmented_bbox = results_dict[class_]
                    measured_bbox = labels_dict[class_]
                    # check if segmented bbox within 30 pixels of real one
                    top_left_diff = (segmented_bbox[0] - measured_bbox[0]) ** 2
                    bottom_right_diff = (segmented_bbox[1] - measured_bbox[1]) ** 2
                    total_diff = np.sum((top_left_diff + bottom_right_diff)) ** 0.5

                    if total_diff > 20:
                        LOGGER.WARNING(f'BBOX for class: {class_} not a good match')

                LOGGER.SUCCESS(f'match for img: {img_path}')
                passes += 1
            else:
                LOGGER.ERROR(f'img_f: {img_path} failed, not all classes match')
                fails += 1

            invid_end_time = round(time.time() - indiv_start_time, 3)
            LOGGER.INFO(f'RUNTIME: {invid_end_time}s\n')

            # UNCOMMENT FOR MORE PRECISE DATA
            # LOGGER.INFO(f'TPR: {round(true_positive_rate*100, 2)}% | FPR: {round(false_positive_rate*100, 2)}% | Accuracy: {round(accuracy, 2)}%\n')
            # data.append([f'Image {i+1}', f'{invid_end_time}s', success, f'{round(false_positive_rate*100, 2)}%', f'{round(true_positive_rate*100, 2)}%', f'{round(accuracy, 2)}%'])

        end_time = round(time.time() - start_time, 3)
        if end_time > 60:
            final_runtime = f'{int(end_time/60)}m {round(end_time%60, 3)}s'
        else:
            final_runtime = f'{end_time}s'

        LOGGER.INFO(f'PASSES: {passes} | FAILS: {fails} | RUNTIME: {final_runtime}')

        # UNCOMMENT FOR MORE PRECISE DATA
        # data.append([f'TOTAL RUNTIME: {final_runtime}'])
        #
        # with open('run_data.csv', 'a', encoding='UTF8', newline='') as f:
        #     writer = csv.writer(f)
        #
        #     if opt == Algorithm.TEMPLATE_MATCHING:
        #         writer.writerow(["Task 2"])
        #     else:
        #         writer.writerow(["Task 3"])
        #     # write the header
        #     writer.writerow(header)
        #     # write multiple rows
        #     writer.writerows(data)
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
