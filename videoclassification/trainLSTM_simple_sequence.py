import numpy as np
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from videoclassification.numSequenceNet import NumSequenceNet

SEQUENCE_LEN = 4


def train_generator(data, labels):
    while True:
        x_train = np.array([np.array(data).transpose()]).transpose() # (3, 4, 1)
        y_train = np.array([np.array(labels).transpose()]).transpose() # (3, 4, 1)
        y_train = to_categorical(y_train) # (3, 4, 4)
        yield x_train, y_train


def train():
    def prepare_data():
        data = np.array([[1, 1, 0, 1], [2, 2, 0, 1], [3, 2, 3, 3], [1, 4, 4, 4]])
        labels = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [0, 0, 0, 0]])
        return data, labels

    data, labels = prepare_data()
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # initialize our VGG-like Convolutional Neural Network
    # trainX: (3, 4), trainY: (3, 4)
    model = NumSequenceNet.build(width=64, height=64, depth=3, sequence_len=SEQUENCE_LEN)

    # initialize our initial learning rate, # of epochs to train for, and batch size
    INIT_LR = 0.01
    EPOCHS = 2
    BS = 32

    # initialize the model and optimizer (you'll want to use binary_crossentropy for 2-class classification)
    print("[INFO] training network...")
    opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    model.fit_generator(train_generator(trainX, trainY), steps_per_epoch=30, epochs=10, verbose=1)


if __name__ == '__main__':
    train()
