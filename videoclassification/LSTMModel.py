from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from sklearn.base import BaseEstimator, ClassifierMixin

from LSTMNetwork import LSTMNetwork
from generators import train_generator, predict_generator

INIT_LR = 0.004
BS = 2


class LSTMModel(BaseEstimator, ClassifierMixin):
    def __init__(
            self, num_of_classes, training_samples, test_samples, sequence_len=9, EPOCHS=None):
        self.num_of_classes = num_of_classes
        self.EPOCHS = EPOCHS
        self.training_samples = training_samples
        self.test_samples = test_samples
        # self.EPOCHS = 500
        self.SEQUENCE_LEN = sequence_len

        self.model = LSTMNetwork.build(
            width=64,
            height=64,
            depth=3,
            sequence_len=self.SEQUENCE_LEN,
            num_of_classes=num_of_classes
        )

        print("[INFO] train data size: " + str(self.training_samples))
        print("[INFO] test data size: " + str(self.test_samples))
        print("[INFO] steps per epoch: " + str(self.training_samples / BS))

        self.tensorboard = TensorBoard(log_dir="logs/{}".format(INIT_LR))

        print("[INFO] training network...")
        opt = SGD(lr=INIT_LR, decay=INIT_LR / num_of_classes)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["categorical_accuracy"])

    def fit(self, trainX, trainY, fit_params):
        print("[INFO] fitting..")
        print(len(trainX))
        print(trainY)
        testX = fit_params['testX']
        testY = fit_params['testY']
        self.label_to_folder = fit_params['label_to_folder']
        self.model.fit_generator(
            generator=train_generator(trainX, trainY, BS, self.num_of_classes, self.label_to_folder),
            steps_per_epoch=self.training_samples / (BS),
            validation_data=train_generator(
                testX,
                testY,
                BS,
                self.num_of_classes,
                self.label_to_folder),
            validation_steps=self.training_samples / BS,
            epochs=self.EPOCHS,
            verbose=1,
            callbacks=[self.tensorboard]
        )

    def predict(self, X):
        print("[INFO] predicting..")
        return self.model.predict_generator(generator=predict_generator(X, num_of_classes=X.shape[0]), steps=X.shape[0])

    def score(self, X, y, **kwargs):
        _, acc = self.model.evaluate_generator(
            generator=train_generator(X, y, BS, self.num_of_classes, self.label_to_folder), steps=1)
        return acc

    def get_model(self):
        return self.model
