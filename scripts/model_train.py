import tensorflow as tf
import numpy as np
from tqdm.keras import TqdmCallback
from model import MyModel

BATCH_SIZE = 64
EPOCHS = 25
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CHANNELS = 3

OPTIMISER = 'Adam'
LOSS = 'binary_crossentropy'

def load_training_data():
    X_train = np.load('training_set.npy') / 255
    y_train = np.load('training_set_labels.npy') / 255
    return X_train, y_train

def get_trained_model(X_train, y_train):
    model = MyModel()
    model.build((1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
    model.freeze_weights()
    model.compile(optimizer = OPTIMISER, loss = LOSS, metrics = ['accuracy'])
    history = model.fit(
        X_train,
        y_train,
        batch_size = BATCH_SIZE,
        steps_per_epoch = BATCH_SIZE / len(X_train),
        epochs = EPOCHS,
        validation_split = 0,
        verbose = 0,
        callbacks = [TqdmCallback(verbose = 0)]
        )
    return model, history

def main():
    X_train, y_train = load_training_data()
    model, history = get_trained_model(X_train, y_train)
    model.save('trained_model', save_format = 'tf')
    return

if __name__ == "__main__":
    main()
