import os
import numpy as  np
import tensorflow as tf
from PIL import Image
import matplotlib.pylab as plt
import cv2 as cv
from cgan import cgan
import random
from typing import List, Tuple
DATA_TRAIN_DIR = 'data/train_set'
DATA_TEST_DIR = 'data/test_set'
MODEL_DIR = 'models/cgan'
DATA_OUTPUT_DIR = 'data/output'

EPOCHS = 3
SAVING_MODEL_FREQUENCY = 1 #how often is the model saved
TESTING_FREQUENCY = 1 #how often is test set processed
BATCH_SIZE = 1
RESTORE_PREVIOUS_MODEL = False
PREVIOUS_MODEl_PATH = None #If you continue training model, provide its path here

def load_batches(data_train_dir: str, data_test_dir: str,
                          batch_size: int) -> Tuple[List, List]:
    batches_train = []
    batches_test = []    
    indices = list(range(int(len(os.listdir(data_train_dir)) / BATCH_SIZE)))
    random.shuffle(indices)
    
    for index in indices:   
        batch_input = np.zeros((batch_size, 256, 256, 3), dtype = np.uint8)
        batch_target = np.zeros((batch_size, 256, 256, 3), dtype = np.uint8)    
        
        load_failed = False
        for j in range(batch_size):
            filename = os.listdir(data_train_dir)[index * BATCH_SIZE + j]
            try:
                im = plt.imread(os.path.join(data_train_dir, filename))
            except:
                print(f"Problem with loading file {filename}")
                load_failed = True
                continue
            im = np.asarray(im, dtype = np.uint8)                
            batch_input[j, :, :, :] = im[:, 0:256, :]
            batch_target[j, :, :, :] = im[:, 256:512, :]
        if not load_failed:
            batches_train.append([batch_input, batch_target])
    
    for filename in os.listdir(data_test_dir):
        try:
            im = plt.imread(os.path.join(data_test_dir, filename))
        except:
            print(f"Problem with loading file {filename}")
            load_failed = True
            continue
        im = np.asarray(im, dtype = np.uint8)         
        batch_input = np.zeros((1, 256, 256, 3), dtype = np.uint8)
        batch_target = np.zeros((1, 256, 256, 3), dtype = np.uint8)
        batch_input[0, :, :, :] = im[:, 0:256, :]
        batch_target[0, :, :, :] = im[:, 256:512, :]
        if not load_failed:
            batches_test.append([batch_input, batch_target])
    return batches_train, batches_test

def preprocess_batch(batch: List) -> Tuple[np.ndarray, np.ndarray]:
    batch_in = np.copy(batch[0])
    batch_target = np.copy(batch[1])
    batch_in = batch_in.astype(np.float32)
    batch_target = batch_target.astype(np.float32)
    batch_in /= 255
    batch_target /= 255    
    if np.random.random() > 0.5:
        batch_in[0, :, :, :] = cv.flip(batch_in[0, :, :, :], 1)
        batch_target[0, :, :, :] = cv.flip(batch_target[0, :, :, :], 1)  
    return batch_in, batch_target


if __name__ == "__main__":    
    input_batch, target_batch, gen_train, discrim_train, outputs = cgan()   
    batches_train, batches_test = load_batches(DATA_TRAIN_DIR, DATA_TEST_DIR, BATCH_SIZE)       
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        if RESTORE_PREVIOUS_MODEL:
            saver.restore(sess, PREVIOUS_MODEl_PATH)
        for epoch in range(EPOCHS):           
            print("Starting epoch", epoch) 
            num_batches_processed = 0        
            for batch in batches_train:
                batch_in, batch_target = preprocess_batch(batch)
                if num_batches_processed % 10 == 0:
                    print(f"Epoch: {epoch} batch: {num_batches_processed}")                     
                num_batches_processed += 1
                
                gen_train.run(feed_dict = {input_batch: batch_in, target_batch: batch_target})
                discrim_train.run(feed_dict = {input_batch: batch_in, target_batch: batch_target})            
            
            if epoch % SAVING_MODEL_FREQUENCY == 0: 
                print("saveing models...")
                saver.save(sess, os.path.join(MODEL_DIR, f"model_{epoch}.ckpt"))
                
            if epoch % TESTING_FREQUENCY == 0:
                i = 0
                print("Processing test images... ")
                for batch in batches_test:                
                    ret = np.zeros((256, 512, 3))
                    batch_in = np.copy(batch[0])
                    batch_in = batch_in.astype(np.float32)
                    batch_in /= 255
                    out = outputs.eval(feed_dict = {input_batch: batch_in})                    
                    ret[:, 0:256, :] = out
                    ret[:, 256:512, :] = batch_in                    
                    ret = np.asarray(ret * 255, dtype = np.uint8)
                    img = Image.fromarray(ret, 'RGB')
                    img.save(os.path.join(DATA_OUTPUT_DIR, f"e{epoch}_{i}.jpg"))
                    i += 1