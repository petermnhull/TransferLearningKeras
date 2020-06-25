import tensorflow as tf
from keras.models import load_model
import numpy as np
from data import load_data

def test_model(model, X_test, y_test):
    scores = model.evaluate(X_test, y_test)
    return scores

def main():
    model = load_model('trained_model')
    _, _, X_test, y_test = load_data()
    scores = test_model(model, X_test, y_test)
    return

if __name__ == "__main__":
    main()
