# Global Variables
BATCH_SIZE = 64
IMAGE_SHAPE = (224, 224, 3)
CLASS_NAMES = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
NUM_CLASSES = len(CLASS_NAMES)
SAVE_HISTORY = True
SAVE_MODEL = True
SAVE_REPORT = True
PRINT_REPORT = True

# Paths
PATH_MODEL = 'trained_model'
PATH_HISTORY = 'history.csv'
PATH_TRAIN = 'training_set'
PATH_TEST = 'test_set'
PATH_DATA = 'data'
PATH_TEST_REPORT = 'model_test.csv'
