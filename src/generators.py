import numpy as np

from keras.utils import to_categorical


# generator: A generator or an instance of `Sequence`
#                 (`keras.utils.Sequence`) object in order to avoid
#                 duplicate data when using multiprocessing.
#                 The output of the generator must be either
#                 - a tuple `(inputs, targets)`
#                 - a tuple `(inputs, targets, sample_weights)`.
#                 This tuple (a single output of the generator) makes a single
#                 batch. Therefore, all arrays in this tuple must have the same
#                 length (equal to the size of this batch). Different batches may
#                 have different sizes. For example, the last batch of the epoch
#                 is commonly smaller than the others, if the size of the dataset
#                 is not divisible by the batch size.
#                 The generator is expected to loop over its data
#                 indefinitely. An epoch finishes when `steps_per_epoch`
#                 batches have been seen by the model.
def train_generator(data, labels, batch_size, num_of_classes, dict):
    while True:
        idxs = np.random.choice(data.shape[0], size=batch_size, replace=False)
        x_train = data[idxs, :]
        y_train = labels[idxs]
        y_train = np.array([np.tile(y_train, (1, 1))]).transpose()
        y_train = to_categorical(y=y_train, num_classes=num_of_classes)
        yield np.expand_dims(x_train, axis=4), y_train.reshape(y_train.shape[0], y_train.shape[2])


def predict_generator(data, num_of_classes):
    i = -1
    while i < num_of_classes - 1:
        i = i + 1
        yield data[[i], 0:9]
