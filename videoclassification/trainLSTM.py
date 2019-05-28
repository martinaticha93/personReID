import pickle
import time

import tensorflow as tf
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.optimizers import SGD

from LSTMNetwork import LSTMNetwork
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
    model = LSTMNetwork.build(width=64, height=64, depth=3, sequence_len=SEQUENCE_LEN, num_of_classes=num_of_classes)
    TRAINING_SAMPLES = len(trainX)
    TEST_SAMPLES = len(testX)

    INIT_LR = 0.004
    EPOCHS = 500
    BS = 30

    print("[INFO] train data size: " + str(TRAINING_SAMPLES))
    print("[INFO] test data size: " + str(TEST_SAMPLES))
    print("[INFO] steps per epoch: " + str(TRAINING_SAMPLES / BS))

    tensorboard = TensorBoard(log_dir="logs/{}".format(INIT_LR))

    print("[INFO] training network...")
    opt = SGD(lr=INIT_LR, decay=INIT_LR / num_of_classes)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["categorical_accuracy"])

    model.fit_generator(
        generator=train_generator(trainX, trainY, SEQUENCE_LEN, BS, num_of_classes, label_to_folder),
        steps_per_epoch=TRAINING_SAMPLES / BS,
        validation_data=train_generator(testX, testY, SEQUENCE_LEN, BS, num_of_classes, label_to_folder),
        validation_steps=TRAINING_SAMPLES / BS,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[tensorboard]
    )

    return model, label_to_folder


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
