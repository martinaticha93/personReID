import pickle

import tensorflow as tf
from keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from videoclassification.datareader import DataReader
from videoclassification.generators import train_generator, predict_generator
from videoclassification.smallVGG import SmallVGGNet

DATA_PATH_TRAIN = "../data/simple_data_set"
SEQUENCE_LEN = 9
MODEL = "model"
LABELS = "labels"
GPU = "4"


def get_generators():
    return ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                              height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                              horizontal_flip=True, fill_mode="nearest")


def train():
    data, labels, num_of_classes, label_to_folder = DataReader.prepare_data(DATA_PATH_TRAIN, SEQUENCE_LEN)
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    model = SmallVGGNet.build(width=64, height=64, depth=3, sequence_len=SEQUENCE_LEN)

    INIT_LR = 0.01
    EPOCHS = 70
    BS = 32

    print("[INFO] training network...")
    opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    model.fit_generator(
        generator=train_generator(trainX, trainY, SEQUENCE_LEN, BS, num_of_classes),
        steps_per_epoch=10,
        epochs=EPOCHS,
        verbose=1
    )

    print("[INFO] evaluating network...")
    predictions = model.predict_generator(
        generator=predict_generator(trainX, batch_size=1, num_of_classes=trainX.shape[0]), steps=trainX.shape[0])
    predictions = predictions.reshape(-1, num_of_classes)
    print(classification_report(trainY, predictions.reshape(-1, 9).argmax(axis=0)))

    return model, label_to_folder


if __name__ == '__main__':
    with tf.device('/gpu:' + GPU):
        model, label_to_folder = train()
        model.save(MODEL)
        f = open(LABELS, "wb")
        f.write(pickle.dumps(label_to_folder))
        f.close()
