import os

import cv2
import numpy as np

DIRECTORY = '/media/martina/Data/School/CTU/thesi/data'
INPUT_FOLDER = 'mars_edges'
OUTPUT_FOLDER = 'mars_edges_selected_20'


def create_folder_from_file_names(file_names: list):
    os.mkdir(os.path.join(DIRECTORY, OUTPUT_FOLDER))
    for file_name in file_names:
        all_folders = os.listdir(os.path.join(DIRECTORY, OUTPUT_FOLDER))
        identity = file_name[0:4]
        if not identity in all_folders:
            os.mkdir(os.path.join(DIRECTORY, OUTPUT_FOLDER, identity))

        original_file_name = file_name.split('_')[0] + '.jpg'
        image = cv2.imread(os.path.join(DIRECTORY, INPUT_FOLDER, identity, original_file_name))
        cv2.imwrite(os.path.join(DIRECTORY, OUTPUT_FOLDER, identity, file_name), image)


if __name__ == '__main__':
    data_names = np.load('/media/martina/Data/School/CTU/thesis/mars_joints/data_names.npy')
    create_folder_from_file_names(data_names)
