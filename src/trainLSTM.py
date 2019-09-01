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
SERVER_MARS_EDGES_KEYPTS_20 = "../data/mars_edges_with_kpts_selected_20_200x200"

LOCAL_MARS_EDGES_20 = '/media/martina/Data/School/CTU/thesis/data/mars_edges_selected_20'
LOCAL_MARS_KEYPTS_20 = '/media/martina/Data/School/CTU/thesis/data/mars_key_points_selected_20'
LOCAL_MARS_EDGES_POSTPRO_20 = '/media/martina/Data/School/CTU/thesis/data/mars_edges_postpro_selected_20'
LOCAL_MARS_EDGES_KEYPTS_20 = '/media/martina/Data/School/CTU/thesis/data/mars_edges_with_kpts_selected_20_200x200'

MARS_EDGES_LOCAL = '/media/martina/Data/School/CTU/thesis/data/mars_joints/joints_edges'
MARS_LOCAL = '/media/martina/Data/School/CTU/thesis/data/mars'

DATA_PATH_TRAIN = SERVER_MARS_EDGES_20
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
            generator=train_generator(x, y, 10, 6, self.label_to_folder), steps=1)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def _train_on_key_points():
    print('[INFO] key points training...')
    print("[INFO] obtaining data...")

    trainX, trainY, testX, testY, num_of_classes, label_to_folder, groups_train = DataReader.prepare_data(
        DATA_PATH_TRAIN, load_key_pts
    )

    model = KeyPtsModel(trainX, trainY, testX, testY, num_of_classes, label_to_folder)
    model.fit()


def _train_on_edges():
    print('[INFO] edges training...')
    print("[INFO] obtaining data...")

    trainX, trainY, testX, testY, num_of_classes, label_to_folder, groups_train = DataReader.prepare_data(
        DATA_PATH_TRAIN, load_edges
    )

    model = EdgesModel(trainX, trainY, testX, testY, num_of_classes, label_to_folder)
    model.fit()


def train():
    if 'key' in DATA_PATH_TRAIN:
        _train_on_key_points()
    elif 'edges' in DATA_PATH_TRAIN:
        _train_on_edges()

    # model = EdgesModel(trainX, trainY, testX, testY, num_of_classes, label_to_folder)
    # model.fit()

    # model = LSTMModel(
    #     num_of_classes=num_of_classes,
    #     training_samples=len(trainX),
    #     test_samples=len(testX)
    # )

    tuned_params = {
        "EPOCHS": [100],
        "INIT_LR": [0.001, 0.0004],
        "DECAY_FACTOR": [0.8]
    }

    # split that is used for cross validation in grid search - for each split there's a run of the alg
    # seems to be useless because this way it splits twice
    # cv = list(GroupShuffleSplit(n_splits=3).split(trainX, trainY, groups_train))
    # gs = GridSearchCV(model, tuned_params, cv=cv)
    # fit_params = {
    #     'label_to_folder': label_to_folder,
    #     'testX': testX,
    #     'testY': testY
    # }
    #
    # gs.fit(trainX, trainY, fit_params=fit_params)

    # print(sorted(gs.cv_results_.keys()))
    # print(gs.best_params_)
    return None, None


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
