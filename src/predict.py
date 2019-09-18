import pickle
import numpy as np

from keras.models import load_model

from generators import predict_generator
from trainLSTM import TEST_X_KEY_POINTS, TEST_Y_KEY_POINTS

DATA_PATH = "../data/simple_data_set_test/0078"
MODEL = "model"
LABELS = "labels"


def predict_on_test_data():
    print("[INFO] loading network and label map...")
    model = load_model(MODEL)
    label_to_folder = pickle.loads(open(LABELS, "rb").read())
    testX = pickle.loads(open(TEST_X_KEY_POINTS, "rb").read())
    testY = pickle.loads(open(TEST_Y_KEY_POINTS, "rb").read())

    print("[INFO] predicting")
    return np.argmax(model.predict_generator(predict_generator(testX), steps=testX.shape[0]), axis=1), testY


if __name__ == '__main__':
    predictions, testY = predict_on_test_data()
    stop = 5

