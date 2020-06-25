import tensorflow as tf
from keras.models import Model
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import pathlib
import numpy as np

dir = os.path.dirname(__file__)

BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 10
