"""
Merge edges and ground truth to dataset, and split to train and test part.
"""
import numpy as np
import os
import cv2 as cv

INPUT_FOLDER_IMAGES = 'data/resized'
INPUT_FOLDER_EDGES = 'data/edges_hed_postpro'
INPUT_FOLDER_MASKS = 'data/masks'
INPUT_FOLDER_BORDER = 'data/borders'
OUTPUT_FOLDER_TRAIN = 'data/train_set'
OUTPUT_FOLDER_TEST = 'data/test_set'

TRAIN_PER_1_TEST = 100 

if __name__ == "__main__":
    num_images_processed = 0
    for filename in os.listdir(INPUT_FOLDER_IMAGES):
        try:
            img = cv.imread(os.path.join(INPUT_FOLDER_IMAGES, filename), 1) 
            edges = cv.imread(os.path.join(INPUT_FOLDER_EDGES, filename), 1) 
            mask = cv.imread(os.path.join(INPUT_FOLDER_MASKS, filename[0:-4] + '.png'), 0) 
            border = cv.imread(os.path.join(INPUT_FOLDER_BORDER, filename), 1)             
                    
            mask = mask.astype(np.bool)
            #skip blank pictures        
            if np.all(mask == False):
                continue
            
            img[mask == False] = 255        
            x = 255 * np.ones((256, 512, 3))
            x[:, 0:256, :] = edges
            x[:, 256:512, :] = img
            
            x2 = 255 * np.ones((256, 512, 3))
            x2[:, 0:256, :] = border        
            x2[:, 256:512, :] = img        
            
            if num_images_processed % TRAIN_PER_1_TEST == 0:            
                cv.imwrite(os.path.join(OUTPUT_FOLDER_TEST, filename), x)
                cv.imwrite(os.path.join(OUTPUT_FOLDER_TEST, 'b_' + filename), x2)
            else:
                cv.imwrite(os.path.join(OUTPUT_FOLDER_TRAIN, filename), x)
                cv.imwrite(os.path.join(OUTPUT_FOLDER_TRAIN, 'b_' + filename), x2)
            
            num_images_processed += 1
            if num_images_processed % 100 == 0:
                print(f"Processed {num_images_processed} / {len(os.listdir(INPUT_FOLDER_IMAGES))} images.")            
        except:
            print(f"Problem with file {filename}")
    
    print(f"Sucessfully processed {num_images_processed} images out of \
    {len(os.listdir(INPUT_FOLDER_IMAGES))} in the input directory.")
          
