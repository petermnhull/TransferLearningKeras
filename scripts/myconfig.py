import os
import tensorflow as tf
from tensorflow import keras

# Directory
dir = os.path.dirname(__file__)

# Set the configuration
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads = 3,
    inter_op_parallelism_threads = 2,
    allow_soft_placement = True,
    device_count = {
        'CPU': 1,
        'GPU' : 0
        }
    )

#Create the session
session = tf.compat.v1.Session(config = config)
tf.compat.v1.keras.backend.set_session(session)

# Global variables
BATCH_SIZE = 1
EPOCHS = 5
IMAGE_SHAPE = (224, 224, 3)
CLASS_NAMES = ['dogs', 'cats']
NUM_CLASSES = len(CLASS_NAMES)
SAVE_HISTORY = True
SAVE_MODEL = True
SAVE_PATH = 'trained_model'
PATH_TRAIN = 'training_set'
PATH_TEST = 'test_set'
