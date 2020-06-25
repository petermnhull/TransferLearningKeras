import tensorflow as tf
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import pathlib
import numpy as np

from model import NewModel

dir = os.path.dirname(__file__)

X_train = np.load('train_set.npy')
y_train = np.load('train_set_labels.npy')

model = NewModel()
model.build((1, 224, 224, 3))
model.freeze_weights()
model.summary()
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#model.fit(X_train, y_train)
