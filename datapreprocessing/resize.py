"""
Resize all images to 256*256 with zero padding.
"""

import math
import os

import cv2 as cv
import numpy as np

INPUT_FOLDER = "data/mars"
OUTPUT_FOLDER = "data/resized"

if __name__ == "__main__":
    num_images_processed = 0
    # for dataset in os.listdir(INPUT_FOLDER):
    dataset = "bbox_train"
    os.mkdir(os.path.join(OUTPUT_FOLDER, dataset))
    try:
        for person in os.listdir(os.path.join(INPUT_FOLDER, dataset)):
            os.mkdir(os.path.join(OUTPUT_FOLDER, dataset, person))
            for file_name in os.listdir(os.path.join(INPUT_FOLDER, dataset, person)):
                rest = 5
                img = cv.imread(os.path.join(INPUT_FOLDER, dataset, person, file_name))
                height = img.shape[0]
                width = img.shape[1]
                result = np.zeros((256, 256, 3))
                if height > width:
                    img = cv.resize(img, (math.floor(256 * width / height), 256))
                    offset = (result.shape[1] - img.shape[1]) // 2
                    result[:, offset:offset + img.shape[1], :] = img
                elif height < width:
                    img = cv.resize(img, (256, math.floor(256 * height / width)))
                    offset = (result.shape[0] - img.shape[0]) // 2
                    result[offset:offset + img.shape[0], :, :] = img
                else:
                    result = cv.resize(img, (256, 256))

                num_images_processed += 1
                cv.imwrite(os.path.join(OUTPUT_FOLDER, dataset, person, file_name), result)
                if num_images_processed % 100 == 0:
                    print(f"resized {num_images_processed} / {len(os.listdir(INPUT_FOLDER))} images.")

    except:
        print("Problem with file: {dataset}")

print(f"Sucessfully resized {num_images_processed} images out of \
    {len(os.listdir(INPUT_FOLDER))} in the input directory.")
