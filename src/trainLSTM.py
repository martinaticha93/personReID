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
SERVER_MARS_KPTS_IMGS_20 = "../data/mars_key_points_selected_20_64x64"

LOCAL_MARS_EDGES_20 = '/media/martina/Data/School/CTU/thesis/data/mars_edges_selected_20_64x64'
LOCAL_MARS_KEYPTS_20 = '/media/martina/Data/School/CTU/thesis/data/mars_key_points_selected_20'
LOCAL_MARS_EDGES_POSTPRO_20 = '/media/martina/Data/School/CTU/thesis/data/mars_edges_postpro_selected_20'
LOCAL_MARS_EDGES_KEYPTS_20 = '/media/martina/Data/School/CTU/thesis/data/mars_edges_with_kpts_selected_20_64x64'

MARS_EDGES_LOCAL = '/media/martina/Data/School/CTU/thesis/data/mars_joints/joints_edges'
MARS_LOCAL = '/media/martina/Data/School/CTU/thesis/data/mars'

DATA_PATH_TRAIN = LOCAL_MARS_KEYPTS_20
MODEL_k = "models/model_k"
MODEL_e = "models/model_e"
MODEL_ke = "models/model_ke"
LABELS = "labels"
TEST_X_KEY_POINTS = 'pickles/testX_k'
TEST_Y_KEY_POINTS = 'pickles/testY_k'
TEST_X_EDGES = 'pickles/testX_e'
TEST_Y_EDGES = 'pickles/testY_e'
TEST_X_EDGES_AND_KPTS = 'pickles/testX_ek'
TEST_Y_EDGES_AND_KPTS = 'pickles/testY_ek'

GPU = "4"


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


def _train_on_key_points(name_of_run):
    print('[INFO] key points training...')
    print("[INFO] obtaining data...")

    trainX, trainY, testX, testY, num_of_classes, label_to_folder, groups_train = DataReader.prepare_data(
        DATA_PATH_TRAIN, load_key_pts
    )

    model = KeyPtsModel(trainX, trainY, testX, testY, num_of_classes, label_to_folder)
    model.fit()
    model.get_model().save(MODEL_k + name_of_run)

def _train_on_kpts_imgs(name_of_run):
    print('[INFO] kpts imgs training...')
    print("[INFO] obtaining data...")

    trainX, trainY, testX, testY, num_of_classes, label_to_folder, groups_train = DataReader.prepare_data(
        DATA_PATH_TRAIN, load_edges
    )

    f = open(TEST_X_EDGES + name_of_run, "wb")
    f.write(pickle.dumps(testX))
    f = open(TEST_Y_EDGES + name_of_run, "wb")
    f.write(pickle.dumps(testY))
    f = open("pickles/label_to_folder_e" + name_of_run, "wb")
    f.write(pickle.dumps(label_to_folder))

    model = EdgesModel(trainX, trainY, testX, testY, num_of_classes, label_to_folder)
    model.fit()
    model.get_model().save(MODEL_e + name_of_run)



def _train_on_edges(name_of_run):
    print('[INFO] edges training...')
    print("[INFO] obtaining data...")

    trainX, trainY, testX, testY, num_of_classes, label_to_folder, groups_train = DataReader.prepare_data(
        DATA_PATH_TRAIN, load_edges
    )

    f = open(TEST_X_EDGES + name_of_run, "wb")
    f.write(pickle.dumps(testX))
    f = open(TEST_Y_EDGES + name_of_run, "wb")
    f.write(pickle.dumps(testY))
    f = open("pickles/label_to_folder_e" + name_of_run, "wb")
    f.write(pickle.dumps(label_to_folder))

    model = EdgesModel(trainX, trainY, testX, testY, num_of_classes, label_to_folder)
    model.fit()
    model.get_model().save(MODEL_e + name_of_run)


def _train_on_edges_and_kpts(name_of_run):
    print('[INFO] edges and key points training...')
    print("[INFO] obtaining data...")

    trainX, trainY, testX, testY, num_of_classes, label_to_folder, groups_train = DataReader.prepare_data(
        DATA_PATH_TRAIN, load_edges
    )

    f = open(TEST_X_EDGES_AND_KPTS + name_of_run, "wb")
    f.write(pickle.dumps(testX))
    f = open(TEST_Y_EDGES_AND_KPTS + name_of_run, "wb")
    f.write(pickle.dumps(testY))
    f = open("pickles/label_to_folder_ke" + name_of_run, "wb")
    f.write(pickle.dumps(label_to_folder))

    model = EdgesModel(trainX, trainY, testX, testY, num_of_classes, label_to_folder)
    model.fit()
    model.get_model().save(MODEL_ke + name_of_run)


def train(name_of_run):
    if 'edges_with_kpts' in DATA_PATH_TRAIN:
        _train_on_edges_and_kpts(name_of_run)
    elif 'key_points_selected_20_64x64' in DATA_PATH_TRAIN:
        _train_on_edges_and_kpts(name_of_run)
    elif 'key' in DATA_PATH_TRAIN:
        _train_on_key_points(name_of_run)
    elif 'edges' in DATA_PATH_TRAIN:
        _train_on_edges(name_of_run)


if __name__ == '__main__':
    start = int(round(time.time()))
    with tf.device('/gpu:' + GPU):

        # DATA_PATH_TRAIN = SERVER_MARS_EDGES_20
        # for i in range(1):
        #     start = int(round(time.time()))
        #     print(f"[INFO] edges training {i}")
        #     train(f"_e_{i}_")
        #     end = int(round(time.time()))
        #     print("[INFO] the training took..." + str(end - start) + "second")
        # print("_______________________________________________________________________________________________________")

        DATA_PATH_TRAIN = SERVER_MARS_EDGES_KEYPTS_20
        for i in range(2):
            start = int(round(time.time()))
            print(f"[INFO] edges keypoints training {i}")
            train(f"_ke_{i}_")
            end = int(round(time.time()))
            print("[INFO] the training took..." + str(end - start) + "second")
        print("_______________________________________________________________________________________________________")

        # DATA_PATH_TRAIN = SERVER_MARS_KPTS_IMGS_20
        # for i in range(5):
        #     start = int(round(time.time()))
        #     print(f"[INFO] kpts imgs training {i}")
        #     train(f"_ki_{i}_")
        #     end = int(round(time.time()))
        #     print("[INFO] the training took..." + str(end - start) + "second")
        # print("_______________________________________________________________________________________________________")

        # DATA_PATH_TRAIN = SERVER_MARS_KEYPTS_20
        # for i in range(4):
        #     start = int(round(time.time()))
        #     print(f"[INFO] key points training {i}")
        #     train(f"_k_{i}_")
        #     end = int(round(time.time()))
        #     print("[INFO] the training took..." + str(end - start) + "second")
        # print("_______________________________________________________________________________________________________")
