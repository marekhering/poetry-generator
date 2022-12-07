from keras import Sequential, Model
from keras.models import load_model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Reshape, Dropout, Activation, Permute, RepeatVector, \
    Lambda, Multiply, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import keras.backend as k


class LSTMModel:
    MODEL_PATH = 'data/models/LSTMModel/weights.h5'
    MODEL_CHECKPOINT = 'data/models/LSTMModel/checkpoint'

    def __init__(
            self,
            load: bool,
            model_path: str = MODEL_PATH,
            total_words: int = None,
            max_sequence_len: int = None
    ):
        self.model: Model = None
        if not load or (self.load(model_path)) is None:
            assert total_words is not None and max_sequence_len is not None
            self.create_with_attention(total_words, 1000, max_sequence_len)

    def create(self, total_words: int, output_dim, max_sequence_len):
        model = Sequential()
        model.add(Embedding(total_words, output_dim, input_length=max_sequence_len - 1))
        model.add(Bidirectional(LSTM(250)))
        model.add(Dense(total_words, activation='softmax'))
        optimizer = Adam(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics='accuracy')
        self.model = model

    def create_with_attention(self, total_words: int, output_dim, max_sequence_len: int, rnn_units: int = 256):
        _in = Input(shape=(None,))
        x = Embedding(total_words, output_dim, input_length=max_sequence_len - 1)(_in)
        x = LSTM(rnn_units, return_sequences=True)(x)
        x = LSTM(rnn_units, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        e = Dense(1, activation='tanh')(x)
        e = Reshape([-1])(e)
        alpha = Activation('softmax')(e)
        alpha_repeated = Permute([2, 1])(RepeatVector(rnn_units)(alpha))
        c = Multiply()([x, alpha_repeated])
        c = Lambda(lambda xin: k.sum(xin, axis=1), output_shape=(rnn_units,))(c)
        _out = Dense(total_words, activation='softmax', name='pitch')(c)
        model = Model([_in], [_out])
        optimizer = Adam(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics='accuracy')
        self.model = model

    def train(self, X, y, epochs: int = 100, batch_size: int = 8):
        model_checkpoint = ModelCheckpoint(
            filepath=self.MODEL_CHECKPOINT,
            save_weights_only=True,
            save_freq=5)
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[model_checkpoint])

    def save(self, _path: str = None):
        self.model.save(_path or self.MODEL_PATH)

    def load(self, _path: str = None):
        try:
            self.model = load_model(_path or self.MODEL_PATH)
            return self.model
        except IOError:
            return None

    def predict(self, X):
        predict_x = self.model.predict([X], verbose=0)
        return np.argmax(predict_x, axis=1)
