import tensorflow as tf
from keras.models import load_model
import numpy as np
import pandas as pd
from data import load_data
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from myconfig import *

def test_model(model, X_test, y_test):
    y_pred = model.predict(
        X_test,
        batch_size = BATCH_SIZE,
        verbose = 1
    )
    y_pred_classes = np.argmax(y_pred, axis = 1)
    print(y_test, '\n', y_pred_classes)
    report = classification_report(y_test, y_pred_classes, target_names = CLASS_NAMES, output_dict = True)
    report_df = pd.DataFrame(report)
    return report_df

def main():
    model = load_model(PATH_MODEL)
    _, _, X_test, y_test = load_data()
    report_df = test_model(model, X_test, y_test)
    if PRINT_REPORT:
        print(report_df)
    if SAVE_REPORT:
        report_df.to_csv(PATH_TEST_REPORT)
    return

if __name__ == "__main__":
    main()
