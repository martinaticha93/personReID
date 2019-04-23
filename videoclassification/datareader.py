import os
import random

import cv2
import numpy as np


class DataReader:

    @staticmethod
    def get_min(data):
        min_len = 1000
        for point in data:
            min_len = min(min_len, len(point))
        return min_len

    @staticmethod
    def prepare_data(data_path, sequence_len):
        print("[INFO] loading images...")
        data = []
        labels = []
        folder_labels_dict = {}

        num_of_folders = -1

        for folder in os.listdir(data_path):
            num_of_folders = num_of_folders + 1
            folder_labels_dict[num_of_folders] = folder
            sequence = []
            directory = os.listdir(os.path.join(data_path, folder))
            directory.sort()
            num_of_imgs_in_sequence = 0
            for file in directory:
                try:
                    image = cv2.imread(os.path.join(data_path, folder, file))
                    image = cv2.resize(image, (64, 64))
                    if file[11:15] == 'F001':
                        num_of_imgs_in_sequence = 0
                        sequence = []
                    if num_of_imgs_in_sequence == sequence_len:
                        data.append(sequence)
                        labels.append(num_of_folders)
                        sequence = []
                        num_of_imgs_in_sequence = 0
                    sequence.append(image)
                    num_of_imgs_in_sequence = num_of_imgs_in_sequence + 1
                except:
                    print("image " + os.path.join(data_path, folder, file) + " could not have been loaded")

            # extract the class label from the image path and update the labels list
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)
        return data, labels
