import pickle
import time

import tensorflow as tf
from keras.callbacks import Callback

from EdgesModel import EdgesModel
from KeyPtsModel import KeyPtsModel
from datareader import DataReader, load_edges, load_key_pts
from generators import train_generator, predict_generator

BBOX_TRAIN = "../data/bbox_train_"
SIMPLE = "../data/simple_data_set"

SERVER_MARS_EDGES_20 = "../data/mars_edges_selected_20"
SERVER_MARS_KEYPTS_20 = "../data/mars_key_points_selected_20"
SERVER_MARS_EDGES_KEYPTS_20 = "../data/mars_edges_with_kpts_selected_20_64x64"

LOCAL_MARS_EDGES_20 = '/media/martina/Data/School/CTU/thesis/data/mars_edges_selected_20'
LOCAL_MARS_KEYPTS_20 = '/media/martina/Data/School/CTU/thesis/data/mars_key_points_selected_20'
LOCAL_MARS_EDGES_POSTPRO_20 = '/media/martina/Data/School/CTU/thesis/data/mars_edges_postpro_selected_20'
LOCAL_MARS_EDGES_KEYPTS_20 = '/media/martina/Data/School/CTU/thesis/data/mars_edges_with_kpts_selected_20_64x64'

MARS_EDGES_LOCAL = '/media/martina/Data/School/CTU/thesis/data/mars_joints/joints_edges'
MARS_LOCAL = '/media/martina/Data/School/CTU/thesis/data/mars'

DATA_PATH_TRAIN = SERVER_MARS_KEYPTS_20
MODEL = "model"
LABELS = "labels"
TEST_X_KEY_POINTS = 'testX_key_points'
TEST_Y_KEY_POINTS = 'testY_key_points'
TEST_X_EDGES = 'testX_edges'
TEST_Y_EDGES = 'testY_edges'
TEST_X_EDGES_AND_KPTS = 'testX_edges_kpts'
TEST_Y_EDGES_AND_KPTS = 'testY_edges_kpts'

GPU = "6"


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
            generator=train_generator(x, y, 10, 6, self.label_to_folder), steps=1)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def _train_on_key_points():
    print('[INFO] key points training...')
    print("[INFO] obtaining data...")

    trainX, trainY, testX, testY, num_of_classes, label_to_folder, groups_train = DataReader.prepare_data(
        DATA_PATH_TRAIN, load_key_pts
    )

    f = open(TEST_X_KEY_POINTS, "wb")
    f.write(pickle.dumps(testX))

    f = open(TEST_Y_KEY_POINTS, "wb")
    f.write(pickle.dumps(testY))

    model = KeyPtsModel(trainX, trainY, testX, testY, num_of_classes, label_to_folder)
    model.fit()
    return model.get_model(), label_to_folder


def _train_on_edges():
    print('[INFO] edges training...')
    print("[INFO] obtaining data...")

    trainX, trainY, testX, testY, num_of_classes, label_to_folder, groups_train = DataReader.prepare_data(
        DATA_PATH_TRAIN, load_edges
    )

    f = open(TEST_X_EDGES, "wb")
    f.write(pickle.dumps(testX))

    f = open(TEST_Y_EDGES, "wb")
    f.write(pickle.dumps(testY))

    model = EdgesModel(trainX, trainY, testX, testY, num_of_classes, label_to_folder)
    model.fit()
    return model.get_model(), label_to_folder

def _train_on_edges_and_kpts():
    print('[INFO] edges and key points training...')
    print("[INFO] obtaining data...")

    trainX, trainY, testX, testY, num_of_classes, label_to_folder, groups_train = DataReader.prepare_data(
        DATA_PATH_TRAIN, load_edges
    )

    f = open(TEST_X_EDGES_AND_KPTS, "wb")
    f.write(pickle.dumps(testX))

    f = open(TEST_Y_EDGES_AND_KPTS, "wb")
    f.write(pickle.dumps(testY))

    model = EdgesModel(trainX, trainY, testX, testY, num_of_classes, label_to_folder)
    model.fit()
    return model.get_model(), label_to_folder


def train():
    if 'edges_with_kpts' in DATA_PATH_TRAIN:
        return _train_on_edges_and_kpts()
    if 'key' in DATA_PATH_TRAIN:
        return _train_on_key_points()
    elif 'edges' in DATA_PATH_TRAIN:
        return _train_on_edges()


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
