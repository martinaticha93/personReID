from keras import Sequential
from keras.layers import BatchNormalization, AveragePooling2D, Flatten, Dense, LSTM


class EdgesNetwork:
    @staticmethod
    def build(num_of_classes):
        print("[INFO] building model...")
        model = Sequential()

        model.add(LSTM(return_sequences=True, dropout=0.2))
        model.add(BatchNormalization())
        model.add(LSTM(return_sequences=True, dropout=0.2))
        model.add(BatchNormalization())
        model.add(LSTM(return_sequences=False, dropout=0.2))
        model.add(BatchNormalization())
        model.add(AveragePooling2D((3, 3), strides=2))
        model.add(Flatten())
        model.add(Dense(
            units=4 * num_of_classes,
            activation='relu',
            kernel_initializer='random_uniform',
            bias_initializer='zeros'))
        model.add(Dense(units=num_of_classes, activation='softmax'))

        # plot_model(model, to_file='model.png', show_shapes=True)
        return model
