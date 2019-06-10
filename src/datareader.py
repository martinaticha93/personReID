import os

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

MAX_NUM_OF_VIDEOS_FOR_IDENTITY = 40
MAX_NUM_OF_VIDEOS_FOR_CAMERA = 4
MIN_NUM_OF_VIDEOS = 4


class DataReader:

    @staticmethod
    def prepare_data(data_path, sequence_len, test_size=0.2):
        print("[INFO] loading images...")
        data = []
        labels = []
        groups_train = []
        label_to_identity = {}
        num_of_identities = -1
        current_camera = ""
        number_of_identical_shots = 0

        for identity in os.listdir(data_path):
            identity_data = {}
            video = []
            directory = os.listdir(os.path.join(data_path, identity))
            directory.sort()
            num_of_imgs_in_video = 0
            num_of_videos_for_camera = 0
            num_of_videos_for_identity = 0
            for file in directory:
                if num_of_videos_for_identity < MAX_NUM_OF_VIDEOS_FOR_IDENTITY:
                    try:
                        image = cv2.imread(os.path.join(data_path, identity, file))
                        image = cv2.resize(image, (64, 64))

                        if file[11:15] == 'F001':
                            num_of_videos_for_camera = 0
                            current_camera = file[6:11]
                            identity_data[current_camera] = []
                            num_of_imgs_in_video = 0
                            video = []

                        if (num_of_videos_for_camera != MAX_NUM_OF_VIDEOS_FOR_CAMERA):

                            if num_of_imgs_in_video == sequence_len:
                                num_of_videos_for_camera = num_of_videos_for_camera + 1
                                identity_data[current_camera].append(video)
                                num_of_videos_for_identity = num_of_videos_for_identity + 1
                                video = []
                                num_of_imgs_in_video = 0

                            video.append(image)
                            num_of_imgs_in_video = num_of_imgs_in_video + 1

                    except:
                        print("image " + os.path.join(data_path, identity, file) + " could not have been loaded")

            num_of_identities = num_of_identities + 1
            label_to_identity[num_of_identities] = identity

            if num_of_videos_for_identity >= MIN_NUM_OF_VIDEOS:

                for i, value in identity_data.items():
                    groups_train.extend([number_of_identical_shots for _ in range(len(value))])
                    number_of_identical_shots = number_of_identical_shots + 1
                    data.extend(value)
                    labels.extend(np.full((len(value)), num_of_identities))

                print("[INFO] loaded identity " + identity)

            else:
                print("[INFO] skipped identity " + identity)

        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        cv = list(GroupShuffleSplit(test_size=test_size, n_splits=1).split(data, labels, groups_train))

        train_indices = cv[0][0]
        test_indices = cv[0][1]

        data_train = data[train_indices]
        data_test = data[test_indices]

        labels_train = labels[train_indices]
        labels_test = labels[test_indices]
        return data_train, labels_train, data_test, labels_test, num_of_identities + 1, label_to_identity, groups_train

    @staticmethod
    def read_test_data(data_path):
        print("[INFO] loading images...")
        data = []

        for file in os.listdir(data_path):
            image = cv2.imread(os.path.join(data_path, file))
            image = cv2.resize(image, (64, 64))

            data.append(image)
        return np.array([data], dtype="float") / 255.0
