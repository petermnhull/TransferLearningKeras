import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD
from keras.utils import to_categorical
from model import MyModel
import numpy as np
import pandas as pd
from data import load_data
from myconfig import *

# Backend
tf.keras.backend.set_floatx('float32')

# Set the configuration
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads = 3,
    inter_op_parallelism_threads = 2,
    allow_soft_placement = True,
    device_count = {
        'CPU' : 1,
        'GPU' : 0
        }
    )

#Create the session
session = tf.compat.v1.Session(config = config)
tf.compat.v1.keras.backend.set_session(session)

def get_model():
    model = MyModel()
    model.build((1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
    model.freeze_weights()
    sgd = SGD(lr = 0.01, momentum = 0, nesterov = False)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def get_trained_model(model, X_train, y_train, epochs = 30, val_split = 0.33):
    train_steps_per_epoch = np.ceil(((len(X_train) * (1 - val_split)) / BATCH_SIZE) - 1)
    val_steps_per_epoch = np.ceil(((len(X_train) * val_split) / BATCH_SIZE) - 1)

    history = model.fit(
        X_train,
        to_categorical(y_train, NUM_CLASSES),
        batch_size = BATCH_SIZE,
        steps_per_epoch = train_steps_per_epoch,
        epochs = epochs,
        validation_split = val_split,
        validation_steps = val_steps_per_epoch,
        verbose = 1
        )

    return model, history

def main():
    X_train, y_train, _, _ = load_data()
    model = get_model()
    model, history = get_trained_model(model, X_train, y_train)
    if SAVE_MODEL:
        model.save(PATH_MODEL, save_format = 'tf')
    if SAVE_HISTORY:
        df = pd.DataFrame(history.history)
        csv_file = 'history.csv'
        with open(csv_file, mode = 'w') as f:
            df.to_csv(f)
    return

if __name__ == "__main__":
    main()
