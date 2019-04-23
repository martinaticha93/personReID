# import the necessary packages
from keras.layers import ConvLSTM2D, Dense, BatchNormalization, Reshape, AveragePooling3D
from keras.models import Sequential
from keras.utils import plot_model


class SmallVGGNet:
    @staticmethod
    def build(width, height, depth, sequence_len):
        print("build")
        # initialize the model along with the input shape to be "channels last" and the channels dimension itself
        input_shape = (sequence_len, height, width, depth)

        model = Sequential()

        model.add(ConvLSTM2D(
            filters=40,
            kernel_size=(3, 3),
            input_shape=(input_shape),
            padding='same',
            return_sequences=True))
        model.add(BatchNormalization())

        model.add(ConvLSTM2D(
            filters=40,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True))
        model.add(BatchNormalization())

        model.add(ConvLSTM2D(
            filters=40,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True))
        model.add(BatchNormalization())

        model.add(AveragePooling3D((1, 64, 64)))
        model.add(Reshape((-1, 40)))
        model.add(Dense(
            units=6,
            activation='sigmoid'))

        plot_model(model, to_file='model.png', show_shapes=True)
        return model
