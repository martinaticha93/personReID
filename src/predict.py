import pickle

import numpy as np
from keras.models import load_model

from generators import predict_generator

DATA_PATH = "../data/simple_data_set_test/0078"
MODEL = "models/model_e_e_0_"
LABELS = "pickles/label_to_folder_e_e_0_"
TEST_X = 'pickles/testX_e_e_0_'
TEST_Y = 'pickles/testY_e_e_0_'


def predict_on_test_data():
    print("[INFO] loading network and label map...")
    model = load_model(MODEL)
    label_to_folder = pickle.loads(open(LABELS, "rb").read())
    testX = pickle.loads(open(TEST_X, "rb").read())
    testY = pickle.loads(open(TEST_Y, "rb").read())

    print("[INFO] predicting")
    return np.argmax(model.predict_generator(predict_generator(testX), steps=testX.shape[0]), axis=1), testY


if __name__ == '__main__':
    predictions, testY = predict_on_test_data()
    stop = 5
