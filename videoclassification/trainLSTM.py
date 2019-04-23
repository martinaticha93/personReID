import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from videoclassification.datareader import DataReader
from videoclassification.smallVGG import SmallVGGNet

DATA_PATH = "/media/martina/Data/School/CTU/thesis/deep-person-reid/data/edges_hed/simple_dataset"
SEQUENCE_LEN = 9


def get_generators():
    return ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                              height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                              horizontal_flip=True, fill_mode="nearest")


def train_generator(data, labels, sequence_len):
    while True:
        # idxs = np.random.randint(data.shape[0], size=2)
        # x_train = data[idxs, :] # (2, 9, 64, 64, 3)
        # y_train = labels[idxs] # (2,)
        x_train = data
        y_train = labels
        y_train = np.array([np.tile(y_train, (sequence_len, 1))]).transpose()
        y_train = to_categorical(y_train)
        yield x_train, y_train


def train():
    # train the network

    data, labels = DataReader.prepare_data(DATA_PATH, SEQUENCE_LEN)
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # initialize our VGG-like Convolutional Neural Network
    model = SmallVGGNet.build(width=64, height=64, depth=3, sequence_len=SEQUENCE_LEN)

    # initialize our initial learning rate, # of epochs to train for, and batch size
    INIT_LR = 0.01
    EPOCHS = 2
    BS = 32

    # initialize the model and optimizer (you'll want to use binary_crossentropy for 2-class classification)
    print("[INFO] training network...")
    opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    model.fit_generator(train_generator(trainX, trainY, SEQUENCE_LEN), steps_per_epoch=30, epochs=10, verbose=1)

    # H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

    # plot the training loss and accuracy
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy (SmallVGGNet)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()
