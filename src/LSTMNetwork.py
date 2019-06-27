from keras.layers import ConvLSTM2D, Dense, BatchNormalization, AveragePooling2D, Flatten
from keras.models import Sequential
from keras.utils import plot_model


class LSTMNetwork:
    @staticmethod
    def build(width, height, depth, sequence_len, num_of_classes):
        print("[INFO] building model...")
        input_shape = (sequence_len, height, width, depth)
        model = Sequential()

        model.add(ConvLSTM2D(
            filters=40,
            kernel_size=(3, 3),
            strides=2,
            input_shape=(input_shape),
            padding='same',
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
            return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(
            filters=60,
            kernel_size=(3, 3),
            padding='same',
            strides=2,
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
            return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(
            filters=80,
            kernel_size=(3, 3),
            padding='same',
            strides=2,
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
            return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(
            filters=100,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
            return_sequences=False))
        model.add(BatchNormalization())
        model.add(AveragePooling2D((3, 3), strides=2))
        model.add(Flatten())
        model.add(Dense(
            units=5 * num_of_classes,
            activation='relu',
            kernel_initializer='random_uniform',
            bias_initializer='zeros'))
        model.add(Dense(
            units=num_of_classes,
            activation='softmax'))

        # plot_model(model, to_file='model.png', show_shapes=True)
        return model
