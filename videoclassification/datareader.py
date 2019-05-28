import math
import os

import cv2
import numpy as np


class DataReader:

    @staticmethod
    def prepare_data(data_path, sequence_len, test_data_percentage=0.2):
        print("[INFO] loading images...")
        data_train = []
        data_test = []
        labels_train = []
        labels_test = []
        label_to_folder = {}
        num_of_folders = -1
        current_camera = ""

        for folder in os.listdir(data_path):
            identity_data = {}
            num_of_sequences_in_folder = 0
            num_of_folders = num_of_folders + 1
            label_to_folder[num_of_folders] = folder
            sequence = []
            directory = os.listdir(os.path.join(data_path, folder))
            directory.sort()
            num_of_imgs_in_sequence = 0
            for file in directory:
                try:
                    image = cv2.imread(os.path.join(data_path, folder, file))
                    image = cv2.resize(image, (64, 64))

                    if file[11:15] == 'F001':
                        current_camera = file[6:11]
                        identity_data[current_camera] = []
                        num_of_imgs_in_sequence = 0
                        sequence = []

                    if num_of_imgs_in_sequence == sequence_len:
                        identity_data[current_camera].append(sequence)
                        num_of_sequences_in_folder = num_of_sequences_in_folder + 1
                        sequence = []
                        num_of_imgs_in_sequence = 0

                    sequence.append(image)
                    num_of_imgs_in_sequence = num_of_imgs_in_sequence + 1


                except:
                    print("image " + os.path.join(data_path, folder, file) + " could not have been loaded")

            test_data_partition = math.ceil(num_of_sequences_in_folder * test_data_percentage)
            added_test_data = 0
            for _, value in identity_data.items():
                if added_test_data < test_data_partition:
                    data_test.extend(value)
                    labels_test.extend(np.full((len(value)), num_of_folders))
                    added_test_data = added_test_data + len(value)
                else:
                    data_train.extend(value)
                    labels_train.extend(np.full((len(value)), num_of_folders))
            print("[INFO] loaded identity " + folder)

        data_train = np.array(data_train, dtype="float") / 255.0
        data_test = np.array(data_test, dtype="float") / 255.0
        labels_train = np.array(labels_train)
        labels_test = np.array(labels_test)

        return data_train, labels_train, data_test, labels_test, num_of_folders + 1, label_to_folder

    @staticmethod
    def read_test_data(data_path):
        print("[INFO] loading images...")
        data = []

        for file in os.listdir(data_path):
            image = cv2.imread(os.path.join(data_path, file))
            image = cv2.resize(image, (64, 64))

            data.append(image)
        return np.array([data], dtype="float") / 255.0
