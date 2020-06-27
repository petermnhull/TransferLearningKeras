import tensorflow as tf
from model import MyModel
from keras.optimizers import Adam
import pandas as pd
import os
from data import load_data
from keras import backend as K

K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads = 2, inter_op_parallelism_threads = 2)))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

BATCH_SIZE = 128
EPOCHS = 20
IMAGE_SHAPE = (224, 224, 3)

SAVE_HISTORY = True
SAVE_MODEL = True
SAVE_PATH = 'trained_model'

def get_trained_model(X_train, y_train):
    model = MyModel()
    model.build((1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
    model.freeze_weights()
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    history = model.fit(
        X_train,
        y_train,
        batch_size = BATCH_SIZE,
        steps_per_epoch = len(X_train) / BATCH_SIZE,
        epochs = EPOCHS,
        validation_split = 0.33,
        verbose = 3
        )
    return model, history

def main():
    X_train, y_train, _, _ = load_data()
    model, history = get_trained_model(X_train, y_train)

    if SAVE_MODEL:
        model.save(SAVE_PATH, save_format = 'tf')

    if SAVE_HISTORY:
        df = pd.DataFrame(history.history)
        csv_file = 'history.csv'
        with open(csv_file, mode = 'w') as f:
            df.to_csv(f)

    return

if __name__ == "__main__":
    main()
