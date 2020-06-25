import tensorflow as tf
import numpy as np
from model import MyModel
from keras.optimizers import Adam
import os

tf.keras.backend.set_floatx('float64')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

BATCH_SIZE = 64
EPOCHS = 20
IMAGE_SHAPE = (224, 224, 3)

SHOW_HISTORY = True
SAVE_MODEL = True

def get_training_data():
    X_train = np.load('training_set.npy') / 255
    y_train = np.load('training_set_labels.npy') / 255
    return X_train, y_train

def get_trained_model(X_train, y_train):
    model = MyModel()
    model.build((1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
    model.freeze_weights()
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    history = model.fit(
        X_train,
        y_train,
        batch_size = BATCH_SIZE,
        steps_per_epoch = BATCH_SIZE / len(X_train),
        epochs = EPOCHS,
        validation_split = 0.2,
        verbose = 2
        )
    return model, history

def show_history(history):
    # Print all information
    print(history.history.keys())
    
    # Accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.show()
    return

def main():
    X_train, y_train = get_training_data()
    model, history = get_trained_model(X_train, y_train)

    if SAVE_MODEL:
        model.save('trained_model', save_format = 'tf')

    if SHOW_HISTORY:
        show_history(history)

    return

if __name__ == "__main__":
    main()
