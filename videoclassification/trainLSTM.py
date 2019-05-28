import pickle
import time

import tensorflow as tf
from keras.callbacks import Callback

from LSTMModel import LSTMModel
from datareader import DataReader
from generators import train_generator, predict_generator

BBOX_TRAIN = "../data/bbox_train_"
SIMPLE = "data/simple_data_set_train"

DATA_PATH_TRAIN = BBOX_TRAIN
SEQUENCE_LEN = 9
MODEL = "model"
LABELS = "labels"
GPU = "7"


class TestCallback(Callback):
    def __init__(self, test_data, label_to_folder):
        self.test_data = test_data
        self.label_to_folder = label_to_folder

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        predictions = self.model.predict_generator(
            generator=predict_generator(x, num_of_classes=x.shape[0]), steps=x.shape[0])
        predictions = predictions.reshape(30, -1)
        loss, acc = self.model.evaluate_generator(
            generator=train_generator(x, y, SEQUENCE_LEN, 10, 6, self.label_to_folder), steps=1)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def train():
    print("[INFO] obtaining data...")
    trainX, trainY, testX, testY, num_of_classes, label_to_folder = DataReader.prepare_data(DATA_PATH_TRAIN,
                                                                                            SEQUENCE_LEN)

    model = LSTMModel(trainX, trainY, testX, testY, num_of_classes, label_to_folder)
    model.fit()

    return model.get_model(), label_to_folder


if __name__ == '__main__':
    start = int(round(time.time()))
    with tf.device('/gpu:' + GPU):
        model, label_to_folder = train()
        end = int(round(time.time()))
        print("[INFO] the training took..." + str(end - start) + "second")
        model.save(MODEL)
        f = open(LABELS, "wb")
        f.write(pickle.dumps(label_to_folder))
        f.close()
