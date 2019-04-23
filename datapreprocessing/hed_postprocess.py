"""
Postprocess the HED edges.
"""
import os

import cv2 as cv
import numpy as np

INPUT_FOLDER = 'data/edges_hed'
OUTPUT_FOLDER = 'data/edges_hed_postpro'
BORDER_FOLDER = 'data/borders'
MASK_FOLDER = 'data/masks'
EDGE_DETECTION_TRESHOLD = 70

if __name__ == "__main__":
    num_images_processed = 0
    for dataset in ['bbox_test', 'bbox_train']:
        persons = os.listdir(os.path.join(INPUT_FOLDER, dataset))
        persons.sort()
        os.mkdir(os.path.join(OUTPUT_FOLDER, dataset))
        try:
            for person in persons:
                    try:
                        os.mkdir(os.path.join(OUTPUT_FOLDER, dataset, person))
                        for file_name in os.listdir(os.path.join(INPUT_FOLDER, dataset, person)):
                            try:
                                img = cv.imread(os.path.join(INPUT_FOLDER, dataset, person, file_name), 0)
                                kernel = np.ones((5, 5), np.float32) / 25
                                img = cv.filter2D(img, -1, kernel)
                                border = 255 * np.ones_like(img)

                                # mask = cv.imread(os.path.join(MASK_FOLDER, filename[0:-4] + '.png'), 0)
                                kernel_2 = np.ones((3, 3), np.float32) / 9
                                # mask = cv.filter2D(mask, -1, kernel_2)
                                # border = ((mask < 255) == (mask > 0))
                                border = 255 * (np.ones_like(border) - border.astype(np.uint8))

                                img[img > EDGE_DETECTION_TRESHOLD] = 255
                                img[img <= EDGE_DETECTION_TRESHOLD] = 0

                                num_images_processed += 1
                                cv.imwrite(os.path.join(OUTPUT_FOLDER, dataset, person, file_name), img)
                                # cv.imwrite(os.path.join(BORDER_FOLDER, filename), border)
                                if num_images_processed % 10 == 0:
                                    print(f"Processed {num_images_processed} / {len(os.listdir(INPUT_FOLDER))} images.")
                            except:
                                print(f"Problem with file: {file_name} person: {person}")
                    except:
                        print(f"Problem with person: {person}")
        except:
            print("error in making directory ")

print(f"Sucessfully processed {num_images_processed} images out of \
    {len(os.listdir(INPUT_FOLDER))} in the input directory.")
