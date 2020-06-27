import tensorflow as tf
from keras.models import load_model
import numpy as np
from data import load_data
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from myconfig import BATCH_SIZE, PATH_MODEL, NUM_CLASSES, CLASS_NAMES

def test_model(model, X_test, y_test):
    y_pred = model.predict(
        X_test,
        batch_size = BATCH_SIZE,
        verbose = 1
    )
    y_pred_classes = np.argmax(y_pred, axis = 1)
    print(y_test, '\n', y_pred_classes)
    report = classification_report(y_test, y_pred_classes, target_names = CLASS_NAMES)
    return report

def main():
    model = load_model(PATH_MODEL)
    _, _, X_test, y_test = load_data()
    report = test_model(model, X_test, y_test)
    print(report)
    return

if __name__ == "__main__":
    main()
