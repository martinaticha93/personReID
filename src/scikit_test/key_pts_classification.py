import numpy as np
import pickle

from sklearn import linear_model
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from datareader import DataReader, load_key_pts
from trainLSTM import LOCAL_MARS_KEYPTS_20


def prepare_data():
    trainX_, trainY, testX_, testY, num_of_classes, label_to_folder, groups_train = DataReader.prepare_data(
        LOCAL_MARS_KEYPTS_20, load_key_pts
    )

    trainX = np.zeros([len(trainY), 34 * 20])

    for i in range(len(trainY)):
        data = trainX_[i, :, :].ravel()
        trainX[i, :] = data

    testX = np.zeros([len(testY), 34 * 20])

    for i in range(len(testY)):
        data = testX_[i, :, :].ravel()
        testX[i, :] = data

    return trainX, trainY, testX, testY, num_of_classes, label_to_folder, groups_train


if __name__ == '__main__':
    trainX, trainY, testX, testY, num_of_classes, label_to_folder, groups_train = prepare_data()

    # with open('trainX.pickle', 'wb') as f:
    #     pickle.dump(trainX, f, pickle.HIGHEST_PROTOCOL)
    # with open('trainY.pickle', 'wb') as f:
    #     pickle.dump(trainY, f, pickle.HIGHEST_PROTOCOL)
    # with open('testX.pickle', 'wb') as f:
    #     pickle.dump(testX, f, pickle.HIGHEST_PROTOCOL)
    # with open('testY.pickle', 'wb') as f:
    #     pickle.dump(testY, f, pickle.HIGHEST_PROTOCOL)

    with open('trainX.pickle', 'rb') as f:
        trainX = pickle.load(f)
    with open('trainY.pickle', 'rb') as f:
        trainY = pickle.load(f)
    with open('testX.pickle', 'rb') as f:
        testX = pickle.load(f)
    with open('testY.pickle', 'rb') as f:
        testY = pickle.load(f)


    print("BernoulliNB")
    clf = BernoulliNB()
    clf.fit(trainX, trainY)
    print(clf.predict(testX))
    print(clf.score(testX, testY))
    print()

    print("DecisionTreeClassifier")
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(trainX, trainY)
    print(clf.predict(testX))
    print(clf.score(testX, testY))

    print("ExtraTreeClassifier")
    clf = ExtraTreeClassifier(random_state=0)
    clf.fit(trainX, trainY)
    print(clf.predict(testX))
    print(clf.score(testX, testY))

    print("GaussianNB")
    clf = GaussianNB()
    clf.fit(trainX, trainY)
    print(clf.predict(testX))
    print(clf.score(testX, testY))



    tt = 4
