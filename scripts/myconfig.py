import os

# Global Variables
BATCH_SIZE = 64
IMAGE_SHAPE = (224, 224, 3)
CLASS_NAMES = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
NUM_CLASSES = len(CLASS_NAMES)
SAVE_HISTORY = True
SAVE_MODEL = True
SAVE_REPORT = True
PRINT_REPORT = True
SHUFFLE = True

# Paths
PATH_MODEL = 'trained_model'
PATH_TRAIN = 'training_set'
PATH_TEST = 'test_set'
PATH_DATA = 'data'

RESULTS_DIR = 'results'
PATH_HISTORY = RESULTS_DIR + '/history.csv'
PATH_TEST_REPORT = RESULTS_DIR + '/test_results.csv'
PATH_ACCURACY = RESULTS_DIR + '/accuracy.png'
PATH_LOSS = RESULTS_DIR + '/loss.png'
