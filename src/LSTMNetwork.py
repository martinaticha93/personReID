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
            name='layer_1',
            filters=40,
            kernel_size=(3, 3),
            strides=2,
            input_shape=(input_shape),
            padding='same',
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
            return_sequences=True))
        model.add(BatchNormalization(
            name='layer_2'
        ))
        model.add(ConvLSTM2D(
            name='layer_3',
            filters=60,
            kernel_size=(3, 3),
            padding='same',
            strides=1,
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
            return_sequences=True))
        model.add(BatchNormalization(
            name='layer_4'
        ))
        model.add(ConvLSTM2D(
            name='layer_5',
            filters=80,
            kernel_size=(2, 2),
            padding='same',
            strides=1,
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
            return_sequences=False))
        model.add(BatchNormalization(
            name='layer_6'
        ))
        model.add(AveragePooling2D
                  ((3, 3),
                   name='layer_7',
                   strides=2))
        model.add(Flatten(
            name='layer_8'
        ))
        model.add(Dense(
            name='layer_9',
            units=4 * num_of_classes,
            activation='relu',
            kernel_initializer='random_uniform',
            bias_initializer='zeros'))
        model.add(Dense(
            name='layer_10',
            units=num_of_classes,
            activation='softmax'))

        plot_model(model, to_file='model.png', show_shapes=True)
        return model
