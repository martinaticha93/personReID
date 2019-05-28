from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from sklearn.base import BaseEstimator, ClassifierMixin

from LSTMNetwork import LSTMNetwork
from generators import train_generator

BBOX_TRAIN = "../data/bbox_train_"
SIMPLE = "../data/simple_data_set_train"

DATA_PATH_TRAIN = SIMPLE
SEQUENCE_LEN = 9
MODEL = "model"
LABELS = "labels"
GPU = "7"


class LSTMModel(BaseEstimator, ClassifierMixin):
    def __init__(self, trainX, trainY, testX, testY, num_of_classes, label_to_folder):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.num_of_classes = num_of_classes
        self.label_to_folder = label_to_folder

        self.model = LSTMNetwork.build(width=64, height=64, depth=3, sequence_len=SEQUENCE_LEN,
                                       num_of_classes=num_of_classes)
        self.TRAINING_SAMPLES = len(trainX)
        self.TEST_SAMPLES = len(testX)

        self.INIT_LR = 0.004
        self.EPOCHS = 500
        self.BS = 30

        print("[INFO] train data size: " + str(self.TRAINING_SAMPLES))
        print("[INFO] test data size: " + str(self.TEST_SAMPLES))
        print("[INFO] steps per epoch: " + str(self.TRAINING_SAMPLES / self.BS))

        self.tensorboard = TensorBoard(log_dir="logs/{}".format(self.INIT_LR))

        print("[INFO] training network...")
        opt = SGD(lr=self.INIT_LR, decay=self.INIT_LR / num_of_classes)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["categorical_accuracy"])

    def fit(self):
        self.model.fit_generator(
            generator=train_generator(self.trainX, self.trainY, SEQUENCE_LEN, self.BS, self.num_of_classes,
                                      self.label_to_folder),
            steps_per_epoch=self.TRAINING_SAMPLES / self.BS,
            validation_data=train_generator(self.testX, self.testY, SEQUENCE_LEN, self.BS, self.num_of_classes,
                                            self.label_to_folder),
            validation_steps=self.TRAINING_SAMPLES / self.BS,
            epochs=self.EPOCHS,
            verbose=1,
            callbacks=[self.tensorboard]
        )

    def get_model(self):
        return self.model
