# import the necessary packages
from keras.layers import LSTM
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential


class NumSequenceNet:
    @staticmethod
    def build(width, height, depth, sequence_len):
        print("build")
        model = Sequential()
        input_shape = (sequence_len, 1)
        model.add(LSTM(32, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(8, return_sequences=True))
        model.add(TimeDistributed(Dense(4, activation='sigmoid')))
        return model
