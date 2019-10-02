from keras import Sequential
from keras.layers import BatchNormalization, Dense, LSTM
from keras.utils import plot_model

from datareader import SEQUENCE_LEN


class KeyPtsNetwork:
    @staticmethod
    def build(num_of_classes):
        print("[INFO] building model...")
        input_shape = (SEQUENCE_LEN, 34)
        model = Sequential()
        model.add((LSTM(input_shape=input_shape, units=34, return_sequences=True, dropout=0.2)))
        model.add(BatchNormalization())
        model.add((LSTM(input_shape=input_shape, units=34, return_sequences=True, dropout=0.2)))
        model.add(BatchNormalization())
        model.add((LSTM(input_shape=input_shape, units=34, return_sequences=False, dropout=0.2)))
        model.add(BatchNormalization())
        model.add(Dense(
            units=34,
            activation='relu',
            kernel_initializer='random_uniform',
            bias_initializer='zeros'))
        model.add(Dense(
            units=34,
            activation='relu',
            kernel_initializer='random_uniform',
            bias_initializer='zeros'))
        model.add(Dense(units=num_of_classes, activation='softmax'))

        # plot_model(model, to_file='model.png', show_shapes=True)
        return model
