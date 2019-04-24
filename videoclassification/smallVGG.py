# import the necessary packages
from keras.layers import ConvLSTM2D, Dense, BatchNormalization, Reshape, AveragePooling3D, AveragePooling2D
from keras.models import Sequential


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
            return_sequences=False))
        model.add(BatchNormalization())

        model.add(AveragePooling2D(( 64, 64)))
        model.add(Reshape((-1, 40)))
        model.add(Dense(
            units=6,
            activation='softmax'))

        # plot_model(model, to_file='model.png', show_shapes=True)
        return model
