import os

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

MAX_NUM_OF_VIDEOS_FOR_IDENTITY = 40
MAX_NUM_OF_VIDEOS_FOR_CAMERA = 4
MIN_NUM_OF_VIDEOS = 4


class DataReader:

    @staticmethod
    # prepares set of videos of "sequence_len" images
    # for each identity, at most MAX_NUM_OF_VIDEOS_FOR_IDENTITY videos are loaded
    # for each identity-camera, at most MAX_NUM_OF_VIDEOS_FOR_CAMERA videos are loaded
    # for each identity, at least MIN_NUM_OF_VIDEOS videos are loaded. Else, the identity is skipped

    # the method returns train + test data together with labels, dict: label -> identity name based on the folder
    # and "groups_train" which is a list of denoting the group of a video. Each group contains videos for unique
    # combination (identity, camera). This list is then used to split data into training a validation test so that
    # videos of the same (identity, camera) combination are not present in both data sets
    def prepare_data(data_path, sequence_len, test_size=0.2):
        print("[INFO] loading images...")
        data = []
        labels = []
        groups = []
        label_to_identity = {}
        num_of_identities = -1
        current_camera = ""
        unique_shots = 0
        identities = os.listdir(data_path)
        identities.sort()

        for identity in identities:
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
                                num_of_videos_for_identity = num_of_videos_for_identity + 1

                                identity_data[current_camera].append(video)
                                video = []
                                num_of_imgs_in_video = 0

                            video.append(image)
                            num_of_imgs_in_video = num_of_imgs_in_video + 1

                    except:
                        print("image " + os.path.join(data_path, identity, file) + " could not have been loaded")


            if num_of_videos_for_identity >= MIN_NUM_OF_VIDEOS:
                num_of_identities = num_of_identities + 1
                label_to_identity[num_of_identities] = identity

                for i, value in identity_data.items():
                    groups.extend([unique_shots for _ in range(len(value))])
                    unique_shots = unique_shots + 1
                    data.extend(value)
                    labels.extend(np.full((len(value)), num_of_identities))

                print("[INFO] loaded identity " + identity)

            else:
                print("[INFO] skipped identity " + identity)

        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        cv = list(GroupShuffleSplit(test_size=test_size, n_splits=1).split(data, labels, groups))

        train_indices = cv[0][0]
        test_indices = cv[0][1]

        data_train = data[train_indices]
        data_test = data[test_indices]

        labels_train = labels[train_indices]
        labels_test = labels[test_indices]
        groups_train = np.array(groups)[train_indices]
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
