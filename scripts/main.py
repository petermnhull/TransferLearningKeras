import tensorflow as tf
from keras.models import load_model
import numpy as np

def get_test_data():
    X_test = np.load('test_set.npy') / 255
    y_test = np.load('test_set_labels.npy') / 255
    return X_test, y_test

def test_model(model, X_test, y_test):
    scores = model.evaluate(X_test, y_test)
    return scores

def main():
    model = load_model('trained_model')
    X_test, y_test = get_test_data()
    scores = test_model(model, X_test, y_test)
    return

if __name__ == "__main__":
    main()
