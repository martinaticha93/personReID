import os

import cv2
import numpy as np

MAX_NUM_OF_VIDEOS_FOR_IDENTITY = 40
MAX_NUM_OF_VIDEOS_FOR_CAMERA = 4
MIN_NUM_OF_VIDEOS = 4


class DataReader:

    @staticmethod
    def prepare_data(data_path, sequence_len, test_data_percentage=0.2):
        print("[INFO] loading images...")
        data_train = []
        data_test = []
        labels_train = []
        labels_test = []
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

            for i, value in identity_data.items():
                groups_train.extend([number_of_identical_shots for _ in range(len(value))])
                number_of_identical_shots = number_of_identical_shots + 1
                data_train.extend(value)
                labels_train.extend(np.full((len(value)), num_of_identities))

            print("[INFO] loaded identity " + identity)

            # test_data_partition = math.ceil(num_of_videos_for_identity * test_data_percentage)
            # added_test_data = 0
            # for i, value in identity_data.items():
            #     if added_test_data < test_data_partition:
            #         data_test.extend(value)
            #         labels_test.extend(np.full((len(value)), num_of_identities))
            #         added_test_data = added_test_data + len(value)
            #     else:
            #         groups_train.extend([number_of_identical_shots for _ in range(len(value))])
            #         number_of_identical_shots = number_of_identical_shots + 1
            #         data_train.extend(value)
            #         labels_train.extend(np.full((len(value)), num_of_identities))
            # print("[INFO] loaded identity " + identity)

        data_train = np.array(data_train, dtype="float") / 255.0
        # data_test = np.array(data_test, dtype="float") / 255.0
        labels_train = np.array(labels_train)
        # labels_test = np.array(labels_test)
        data_test = np.array([])
        labels_test = np.array([])

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
