"""
Uses Holistic Edge Detector (HED) to extract edges from input images.
"""

import os
import sys
from typing import Tuple

import cv2 as cv
import numpy as np
from PIL import Image

INPUT_FOLDER = 'data/resized'
MASK_FOLDER = 'data/masks'
OUTPUT_FOLDER = 'data/edges_hed'
USING_GPU = False
GPU_ID = 1

caffe_root = 'custom_caffe/'
try:
    sys.path.remove(caffe_root + 'caffe')
except:
    pass
sys.path.insert(0, caffe_root + 'caffe')
# importing custom extension to Caffe. It must be done after standart Caffe is imported.
import caffe

sys.path.remove(caffe_root + 'caffe')


def load_and_preprocess_input(input_folder: str, filename: str) -> Tuple[np.ndarray, int]:
    image = Image.open(os.path.join(input_folder, filename))
    image = np.array(image, dtype=np.float32)
    image_width = image.shape[1]
    # mask = Image.open(os.path.join(MASK_FOLDER, filename[0:-4] + ".png"))
    # mask = np.array(mask, dtype=np.float32)
    # image[mask == 0] = 255
    # Substract the mean of training data on which HED was trained:
    mean_of_hed_train_set = 110
    image -= mean_of_hed_train_set
    inp = np.zeros((500, 500, 3))
    offset = 100
    # we are adding offset to fix a bug in HED implementation (the output image is shifted)
    inp[offset:offset + image.shape[0], offset:offset + image.shape[1], :] = image
    return inp, image_width


def pass_through_hed(inp: np.ndarray) -> np.ndarray:
    net.blobs['data'].data[...][0][0] = inp[:, :, 0]
    net.blobs['data'].data[...][0][1] = inp[:, :, 1]
    net.blobs['data'].data[...][0][2] = inp[:, :, 2]
    net.forward()

    output_layers = [net.blobs['sigmoid-dsn1'].data[0][0, :, :],
                     net.blobs['sigmoid-dsn2'].data[0][0, :, :],
                     net.blobs['sigmoid-dsn3'].data[0][0, :, :],
                     net.blobs['sigmoid-dsn4'].data[0][0, :, :],
                     net.blobs['sigmoid-dsn5'].data[0][0, :, :],
                     net.blobs['sigmoid-fuse'].data[0][0, :, :], ]

    return output_layers[2]


def postprocess_edges(edges: np.ndarray, image_width: int) -> np.ndarray:
    edges = 1 - edges
    edges *= 255
    edges = edges[138:138 + image_width, 138:138 + image_width]
    edges = cv.resize(edges, (256, 256))
    return Image.fromarray(edges.astype(np.uint8))


if __name__ == "__main__":
    if USING_GPU:
        caffe.set_mode_gpu()
        caffe.set_device(GPU_ID)
    else:
        caffe.set_mode_cpu()
    model_root = 'models/hed/'
    net = caffe.Net(os.path.join(model_root, 'hed_deploy.prototxt'), caffe.TEST)

    num_images_processed = 0
    # for dataset in os.listdir(INPUT_FOLDER):
    dataset = "bbox_test"
    # os.mkdir(os.path.join(OUTPUT_FOLDER, dataset))
    persons = os.listdir(os.path.join(INPUT_FOLDER, dataset))
    persons.sort()
    try:
        for person in persons:
            if person > '1128':
                try:
                    os.mkdir(os.path.join(OUTPUT_FOLDER, dataset, person))
                    for file_name in os.listdir(os.path.join(INPUT_FOLDER, dataset, person)):
                        try:
                            inp, image_width = load_and_preprocess_input(os.path.join(INPUT_FOLDER, dataset, person),
                                                                         file_name)
                            edges = pass_through_hed(inp)
                            result = postprocess_edges(edges, image_width)
                            result.save(os.path.join(OUTPUT_FOLDER, dataset, person, file_name))
                            num_images_processed += 1
                            if num_images_processed % 100 == 0:
                                print("Processed" + str(num_images_processed) + " images.")
                        except:
                            print("could not transform: " + file_name)
                except:
                    print("error in making directory person " + person)

    except:
        print("problem with loading or generating mask from file name: " + num_images_processed)

# print(f"Sucessfully processed {num_images_processed} images out \
#     of {len(os.listdir(INPUT_FOLDER))} in the input directory.")
