"""
For images in input folder, find mask where ceratin class (e.g. car) is present.
"""
import os
import sys

import cv2 as cv
import numpy as np

CLASS = 'car'
INPUT_FOLDER = 'data/resized'
OUTPUT_FOLDER = 'data/masks'

root_dir = os.path.abspath("models/mask-rcnn")
sys.path.append(root_dir)
from mrcnn import utils
import mrcnn.model as modellib

sys.path.append(os.path.join(root_dir, "coco/"))
import coco


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


if __name__ == "__main__":
    config = InferenceConfig()

    class_names = ['BG', ' on', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']
    class_num = class_names.index(CLASS)

    model_dir = os.path.join(root_dir, "logs")
    coco_model_path = os.path.join(root_dir, "mask_rcnn_coco.h5")
    if not os.path.exists(coco_model_path):
        print("Downloading model weights...")
        utils.download_trained_weights(coco_model_path)

    model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
    model.load_weights(coco_model_path, by_name=True)

    num_images_processed = 0
    for filename in os.listdir(INPUT_FOLDER):
        try:
            image = cv.imread(os.path.join(INPUT_FOLDER, filename))
            results = model.detect([image], verbose=0)
            r = results[0]
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.bool)
            for i in range(len(r['class_ids'])):
                if r['class_ids'][i] == class_num:
                    mask += r['masks'][:, :, i]
            cv.imwrite(os.path.join(OUTPUT_FOLDER, filename[:0 - 4] + ".png"), 255 * mask.astype(np.uint8))
            num_images_processed += 1
            if num_images_processed % 10 == 0:
                print(f"Processed {num_images_processed} / {len(os.listdir(INPUT_FOLDER))} images.")
        except:
            pass

    print(f"Sucessfully processed {num_images_processed} images out of \
    {len(os.listdir(INPUT_FOLDER))} in the input directory.")
