from keras import Sequential
from keras.layers import BatchNormalization, Dense, LSTM, ConvLSTM2D, Flatten, AveragePooling2D

from datareader import SEQUENCE_LEN


class KeyPtsNetwork:
    @staticmethod
    def build(num_of_classes):
        print("[INFO] building model...")
        input_shape = (SEQUENCE_LEN, 51)
        model = Sequential()

        model.add((LSTM(
            input_shape=input_shape,
            units=20,
            return_sequences=True,
            dropout=0.2
        )))
        model.add(BatchNormalization())
        model.add((LSTM(
            input_shape=input_shape,
            units=20,
            return_sequences=True,
            dropout=0.2
        )))
        model.add(BatchNormalization())
        model.add((LSTM(
            input_shape=input_shape,
            units=20,
            return_sequences=False,
            dropout=0.2
        )))
        model.add(BatchNormalization())
        # model.add(AveragePooling2D(strides=2))
        # model.add(Flatten())
        model.add(Dense(
            units=4 * num_of_classes,
            activation='relu',
            kernel_initializer='random_uniform',
            bias_initializer='zeros'))
        model.add(Dense(units=num_of_classes, activation='softmax'))

        # plot_model(model, to_file='model.png', show_shapes=True)
        return model
