import os

import cv2
import numpy as np
import scipy.misc

DIRECTORY = '/media/martina/Data/School/CTU/thesis/data'
INPUT_FOLDER = 'mars_edges_with_kpts_selected_20_200x200'
OUTPUT_FOLDER = 'mars_edges_with_kpts_selected_20_200x200'

def clean_third_dimension_of_images(data_path):
    directory = os.listdir(data_path)
    directory.sort()
    for folder in directory:
        print(f'identity ${folder}')
        print(folder)
        directory_2 = os.listdir(os.path.join(data_path, folder))
        directory_2.sort()
        for file_name in directory_2:
            image = cv2.imread(os.path.join(data_path, folder, file_name))
            image = cv2.resize(image, (150, 150))
            scipy.misc.imsave(os.path.join(data_path, folder, file_name), image)


def read_and_write_img(data_path_in, data_path_out):
    image = cv2.imread(data_path_in)
    cv2.imwrite(data_path_out, image)


def read_and_write_key_pts(data_path_in, data_path_out):
    _, keypoint_set = np.load(os.path.join(data_path_in + '.npy'))
    np.save(data_path_out, keypoint_set)

def create_folder_from_file_names(file_names: list):
    os.mkdir(os.path.join(DIRECTORY, OUTPUT_FOLDER))
    for file_name in file_names:
        all_folders = os.listdir(os.path.join(DIRECTORY, OUTPUT_FOLDER))
        identity = file_name[0:4]
        if not identity in all_folders:
            os.mkdir(os.path.join(DIRECTORY, OUTPUT_FOLDER, identity))

        original_file_name = file_name.split('_')[0] + '.jpg'

        read_and_write_key_pts(
            os.path.join(DIRECTORY, INPUT_FOLDER, identity, original_file_name),
            os.path.join(DIRECTORY, OUTPUT_FOLDER, identity, file_name)
        )


if __name__ == '__main__':
    clean_third_dimension_of_images(os.path.join(DIRECTORY, INPUT_FOLDER))
    # data_names = np.load('/media/martina/Data/School/CTU/thesis/data/data_names.npy')
    # create_folder_from_file_names(data_names)
