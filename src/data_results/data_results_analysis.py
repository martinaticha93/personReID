LOCAL_MARS_EDGES_20 = '/media/martina/Data/School/CTU/thesis/data/mars_edges_selected_20_64x64'
PREDICTION_GROUPS = '/media/martina/Data/School/CTU/thesis/data/prediction_groups'
identitiesToPredictidValues_dir = '../pickles/identitiesToPredictidValues'

import os
import pickle

import cv2
import scipy.misc


def create_groups():
    os.makedirs(PREDICTION_GROUPS)
    identitiesToPredictidValues = pickle.loads(open(identitiesToPredictidValues_dir, "rb").read())
    for key, value in identitiesToPredictidValues.items():
        os.makedirs(os.path.join(PREDICTION_GROUPS, key))
        imgs = os.listdir(os.path.join(LOCAL_MARS_EDGES_20, key))[0:20]
        for i in range(len(imgs)):
            img = cv2.imread(os.path.join(LOCAL_MARS_EDGES_20, key, imgs[i]))
            scipy.misc.imsave(
                os.path.join(os.path.join(PREDICTION_GROUPS, key, "original_" + str(i) + ".jpg")), img)
        for prediction in value:
            imgs = os.listdir(os.path.join(LOCAL_MARS_EDGES_20, prediction))[0:20]
            for i in range(len(imgs)):
                img = cv2.imread(os.path.join(LOCAL_MARS_EDGES_20, prediction, imgs[i]))
                scipy.misc.imsave(
                    os.path.join(os.path.join(PREDICTION_GROUPS, key, prediction + "_" + str(i) + ".jpg")), img)


if __name__ == '__main__':
    create_groups()
