import json
import os
from typing import List

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

MAX_NUM_OF_VIDEOS_FOR_IDENTITY = 10
MAX_NUM_OF_VIDEOS_FOR_CAMERA = 4
MIN_NUM_OF_VIDEOS = 4


# sorts by score and limits num of videos for camera
def _get_best_sequences(videos: list):
    videos = sorted(videos, key=lambda i: i['score'], reverse=True)
    return videos[0:MAX_NUM_OF_VIDEOS_FOR_CAMERA]


# count of all videos no matter which camera they come from
def _get_num_of_videos_in_dict(identity_data: dict):
    num_of_videos = 0
    for videos in identity_data.values():
        num_of_videos = num_of_videos + len(videos)
    return num_of_videos


# copy of a dictionary without values
def _get_dict_with_identical_keys(identity_data: dict):
    result = {}
    for key in identity_data.keys():
        result[key] = []
    return result


# selects videos with highest possible scores optimally distributed among all cameras with limitation on num of videos
# per camera and num of videos per identity
def _select_identity_data(identity_data: dict):
    for camera, videos in identity_data.items():
        identity_data[camera] = _get_best_sequences(videos)

    if _get_num_of_videos_in_dict(identity_data) <= MAX_NUM_OF_VIDEOS_FOR_IDENTITY:
        return identity_data

    num_of_selected_videos = 0
    level = 0
    selected_videos = _get_dict_with_identical_keys(identity_data)

    while True:
        for camera, videos in identity_data.items():
            if len(videos) > level:
                selected_videos[camera].append(videos[level]['video'])
                num_of_selected_videos = num_of_selected_videos + 1
                if num_of_selected_videos == MAX_NUM_OF_VIDEOS_FOR_IDENTITY:
                    break
        if num_of_selected_videos == MAX_NUM_OF_VIDEOS_FOR_IDENTITY:
            break

    return selected_videos


def _get_video_score(video: List):
    score = 0
    for img in video:
        score = score + img['score']
    return score


def _add_identity(data, labels, groups, identity_data, unique_cameras, num_of_identities):
    for camera, videos in identity_data.items():
        groups.extend([unique_cameras for _ in range(len(videos))])
        unique_cameras = unique_cameras + 1
        data.extend(videos)
        labels.extend(np.full((len(videos)), num_of_identities))
    return data, labels, groups, unique_cameras


def _get_img_score(image_name):
    name_split = image_name.split('_')
    if len(name_split) > 1:
        return float(name_split[1][:-4])
    return 0


def _load_one_identity(data_path, identity, sequence_len):
    identity_data = {}
    video = []
    directory = os.listdir(os.path.join(data_path, identity))
    directory.sort()
    num_of_imgs_in_video = 0
    num_of_videos_for_identity = 0
    current_camera = ''

    for i, file in enumerate(directory):
        if i == 0:
            video = []
            num_of_imgs_in_video = 0
            current_camera = file[6:11]
            identity_data[current_camera] = []

        if (i > 0):
            file_name = int(file[12:15])
            previous_file_name = int(directory[i - 1][12:15])
            if (file_name - 1 != previous_file_name):
                video = []
                num_of_imgs_in_video = 0
                if (current_camera != file[6:11]):
                    current_camera = file[6:11]
                    identity_data[current_camera] = []
        try:
            image = cv2.imread(os.path.join(data_path, identity, file))
            image = cv2.resize(image, (64, 64))

            if num_of_imgs_in_video == sequence_len:
                score = _get_video_score(video=video)
                video = [image for image in video]  # drop score
                identity_data[current_camera].append({'score': score, 'video': video})
                video = []
                num_of_imgs_in_video = 0
                num_of_videos_for_identity = num_of_videos_for_identity + 1

            image_score = _get_img_score(file)
            video.append({'image': image, 'score': image_score, 'file_name': file})
            num_of_imgs_in_video = num_of_imgs_in_video + 1

        except:
            print("image " + os.path.join(data_path, identity, file) + " could not have been loaded")

    return num_of_videos_for_identity, identity_data


class DataReader:

    @staticmethod
    # prepares set of videos of "sequence_len" images
    # for each identity, at most MAX_NUM_OF_VIDEOS_FOR_IDENTITY videos are loaded
    # for each identity-camera, at most MAX_NUM_OF_VIDEOS_FOR_CAMERA videos are loaded
    # for each identity, at least MIN_NUM_OF_VIDEOS videos are loaded. Else, the identity is skipped

    # the method returns train + test data together with labels, dict: label -> identity name based on the folder
    # and "groups_train" which is a list denoting the group of a video. Each group contains videos for unique
    # combination (identity, camera). This list is then used to split data into training a validation test so that
    # videos of the same (identity, camera) combination are not present in both data sets
    def prepare_data(data_path, sequence_len, test_size=0.2):
        print("[INFO] loading images...")

        def _videos_to_img_key(video: list, key: str):
            return [img[key] for img in video]

        data = []
        labels = []
        groups = []
        label_to_identity = {}
        num_of_identities = -1
        unique_cameras = 0
        identities = os.listdir(data_path)
        identities.sort()

        for identity in identities:
            num_of_videos_for_identity, identity_data = _load_one_identity(data_path, identity, sequence_len)

            if num_of_videos_for_identity >= MIN_NUM_OF_VIDEOS:
                num_of_identities = num_of_identities + 1
                label_to_identity[num_of_identities] = identity
                identity_data = _select_identity_data(identity_data)
                data, labels, groups, unique_cameras = _add_identity(data,
                                                                     labels,
                                                                     groups,
                                                                     identity_data,
                                                                     unique_cameras,
                                                                     num_of_identities)
                print("[INFO] loaded identity " + identity)
            else:
                print("[INFO] skipped identity " + identity)

        data_names = [_videos_to_img_key(video, key='file_name') for video in data]
        data_names_file = json.dumps(data_names)
        f = open("data_names.json", "w")
        f.write(data_names_file)
        f.close()
        # with open('data_names.json') as json_file:
        #     data_names_file = json.load(json_file)

        data = [_videos_to_img_key(video, key='image') for video in data]
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
