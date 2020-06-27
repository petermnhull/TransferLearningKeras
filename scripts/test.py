import tensorflow as tf
from keras.models import load_model
import numpy as np
from data import load_data
from keras.utils import to_categorical
from myconfig import BATCH_SIZE, PATH_MODEL, NUM_CLASSES

def test_model(model, X_test, y_test):
    scores = model.evaluate(
        X_test,
        to_categorical(y_test, NUM_CLASSES),
        batch_size = BATCH_SIZE
        )
    return scores

def main():
    model = load_model(PATH_MODEL)
    _, _, X_test, y_test = load_data()
    scores = test_model(model, X_test, y_test)
    print(scores)
    return

if __name__ == "__main__":
    main()
