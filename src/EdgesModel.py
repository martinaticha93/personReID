from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from sklearn.base import BaseEstimator, ClassifierMixin

from EdgesNetwork import EdgesNetwork
from generators import train_generator, predict_generator


class EdgesModel(BaseEstimator, ClassifierMixin):
    def __init__(self, trainX, trainY, testX, testY, num_of_classes, label_to_folder):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX

        img_width = testX[0,0].shape[0]

        self.testY = testY

        self.num_of_classes = num_of_classes
        self.label_to_folder = label_to_folder

        self.model = EdgesNetwork.build(num_of_classes, img_width)
        self.TRAINING_SAMPLES = len(trainX)
        self.TEST_SAMPLES = len(testX)

        self.INIT_LR = 0.005
        self.EPOCHS = 2
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
            generator=train_generator(self.trainX, self.trainY, self.BS, self.num_of_classes,
                                      self.label_to_folder),
            steps_per_epoch=self.TRAINING_SAMPLES / self.BS,
            validation_data=train_generator(self.testX, self.testY, self.BS, self.num_of_classes,
                                            self.label_to_folder),
            validation_steps=self.TRAINING_SAMPLES / self.BS,
            epochs=self.EPOCHS,
            verbose=1,
            callbacks=[self.tensorboard]
        )

    def predict(self, X):
        print("[INFO] predicting..")
        return self.model.predict_generator(generator=predict_generator(X, num_of_classes=X.shape[0]), steps=X.shape[0])

    def score(self, X, y, **kwargs):
        _, acc = self.model.evaluate_generator(
            generator=train_generator(X, y, self.BS, self.num_of_classes, self.label_to_folder), steps=1)
        print("[INFO] score.. " + str(acc))
        print("[INFO] X len.. " + str(len(X)))
        return acc

    def get_model(self):
        return self.model
