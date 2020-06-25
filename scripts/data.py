import os
import numpy as np
from skimage.io import imread
from pathlib import Path
import cv2

dir = os.path.dirname(__file__)
PATH_TRAIN = 'training_set'
PATH_TEST = 'test_set'

NUM_CLASSES = 2
IMAGE_SHAPE = (224, 224, 3)
CLASS_NAMES = ['dogs', 'cats']

def get_data():
    for dataset in [PATH_TRAIN, PATH_TEST]:
        # Initialise
        images = []
        labels = []
        for i in range(len(CLASS_NAMES)):
            # Get paths
            path = os.path.join(dir, 'data', dataset, CLASS_NAMES[i])
            paths = Path(path).glob('**/*.jpg')
            for path in paths:
                # Process image
                image = imread(path)
                image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]), IMAGE_SHAPE[2])
                images.append(image)
                # Process label
                labels.append(i)

                print('Done: ', path)

        # Export arrays to file
        np.save(dataset, images)
        np.save(dataset + '_labels', labels)

    return

def main():
    get_data()
    return

if __name__ == "__main__":
    main()
