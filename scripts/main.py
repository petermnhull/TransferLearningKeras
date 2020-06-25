import tensorflow as tf
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import pathlib
import numpy as np
from tqdm.keras import TqdmCallback
from model import NewModel

dir = os.path.dirname(__file__)

BATCH_SIZE = 64

X_train = np.load('training_set.npy')
y_train = np.load('training_set_labels.npy')
X_test = np.load('test_set.npy')
y_test = np.load('test_set_labels.npy')

def get_trained_model():
    model = NewModel()
    model.build((1, 224, 224, 3))
    model.freeze_weights()
    model.summary()
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(
        X_train,
        y_train,
        batch_size = BATCH_SIZE,
        steps_per_epoch = BATCH_SIZE / len(X_train),
        epochs = 20,
        validation_split = 0,
        verbose = 0,
        callbacks = [TqdmCallback(verbose = 1)]
        )
    return model

def test_model(model):
    scores = model.evaluate(X_test, y_test)
    return scores

def main():
    model = get_trained_model()
    scores = test_model(model)
    return

if __name__ == "__main__":
    main()
