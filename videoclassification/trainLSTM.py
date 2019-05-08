import pickle

from keras.callbacks import Callback
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

from datareader import DataReader
from generators import train_generator
from LSTMNetwork import LSTMNetwork

DATA_PATH_TRAIN = "../data/simple_data_set_train"
SEQUENCE_LEN = 9
MODEL = "model"
LABELS = "labels"


class TestCallback(Callback):
    def __init__(self, test_data, label_to_folder):
        self.test_data = test_data
        self.label_to_folder = label_to_folder

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        # predictions = self.model.predict_generator(
        #     generator=predict_generator(x, num_of_classes=x.shape[0]), steps=x.shape[0])
        # predictions = predictions.reshape(30, -1)
        loss, acc = self.model.evaluate_generator(
            generator=train_generator(x, y, SEQUENCE_LEN, 10, 6, self.label_to_folder), steps=1)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def train():
    print("[INFO] obtaining data...")
    data, labels, num_of_classes, label_to_folder = DataReader.prepare_data(DATA_PATH_TRAIN, SEQUENCE_LEN)
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.00, random_state=42)
    model = LSTMNetwork.build(width=64, height=64, depth=3, sequence_len=SEQUENCE_LEN)

    INIT_LR = 0.1
    EPOCHS = 100
    BS = 30

    print("[INFO] training network...")
    opt = SGD(lr=INIT_LR, decay=INIT_LR / 100)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["categorical_accuracy"])

    model.fit_generator(
        generator=train_generator(trainX, trainY, SEQUENCE_LEN, BS, num_of_classes, label_to_folder),
        steps_per_epoch=10,
        validation_data=train_generator(trainX, trainY, SEQUENCE_LEN, BS, num_of_classes, label_to_folder),
        validation_steps=10,
        epochs=EPOCHS,
        verbose=1
    )

    return model, label_to_folder


if __name__ == '__main__':
    model, label_to_folder = train()
    model.save(MODEL)
    f = open(LABELS, "wb")
    f.write(pickle.dumps(label_to_folder))
    f.close()
