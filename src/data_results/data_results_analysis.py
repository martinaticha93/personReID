LOCAL_MARS_EDGES_20 = '/media/martina/Data/School/CTU/thesis/data/mars_edges_selected_20_64x64'
PREDICTION_GROUPS = '/media/martina/Data/School/CTU/thesis/data/prediction_groups'
identitiesToPredictidValues_dir = '../pickles/identitiesToPredictidValues'
CSV_OUTPUTS_DIR = '../csv_outputs'

import os
import pickle

import cv2
import numpy as np
import pandas as pd
import scipy.misc


# loads the dictionary of identities to predicted values and creates a folder containing a folder for each identity
# from the test set where there is one video sequence for this identity and for each identity that was confused with
# this identity
# to generate the dictionary, use notebooks/predictions
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


# stores the confusion metrix into csv file
# the input parameter is the dictionary same as in the previous function
def createConfusionMatrix(dict_of_predictions: dict):
    list_of_lists = [value for value in dict_of_predictions.values()]
    keys = list(set([item for sublist in list_of_lists for item in sublist] + list(dict_of_predictions.keys())))
    keys.sort()
    my_array = np.zeros([len(keys), len(keys)])
    confusion_matrix = pd.DataFrame(my_array)
    confusion_matrix.columns = keys
    confusion_matrix = confusion_matrix.set_index(pd.Index(list(keys)))

    for key, values in dict_of_predictions.items():
        for value in values:
            confusion_matrix[key][value] = confusion_matrix[key][value] + 1
    confusion_matrix.to_csv(os.path.join(CSV_OUTPUTS_DIR, 'confusion_matrix.csv'))


if __name__ == '__main__':
    createConfusionMatrix(pickle.loads(open(identitiesToPredictidValues_dir, "rb").read()))
    # create_groups()
