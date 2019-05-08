# import the necessary packages
from keras.layers import ConvLSTM2D, Dense, BatchNormalization, Reshape, AveragePooling2D
from keras.models import Sequential


# from keras.utils import plot_model


class SmallVGGNet:
    @staticmethod
    def build(width, height, depth, sequence_len):
        print("build")
        input_shape = (sequence_len, height, width, depth)

        model = Sequential()

        model.add(ConvLSTM2D(
            filters=40,
            kernel_size=(3, 3),
            input_shape=(input_shape),
            padding='same',
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
            return_sequences=True))
        model.add(BatchNormalization())

        model.add(ConvLSTM2D(
            filters=40,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
            return_sequences=True))
        model.add(BatchNormalization())

        model.add(ConvLSTM2D(
            filters=40,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
            return_sequences=False))
        model.add(BatchNormalization())

        model.add(AveragePooling2D((64, 64)))
        model.add(Reshape((-1, 40)))
        model.add(Dense(
            units=20,
            activation='relu',
            kernel_initializer='random_uniform',
            bias_initializer='zeros'))
        model.add(Dense(
            units=6,
            activation='softmax'))

        # plot_model(model, to_file='model.png', show_shapes=True)
        return model
