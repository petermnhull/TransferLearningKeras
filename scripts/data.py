import os
import numpy as np
from skimage.io import imread
from pathlib import Path
import cv2

from myconfig import dir, CLASS_NAMES, IMAGE_SHAPE, PATH_TRAIN, PATH_TEST

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
        np.save(dataset, np.asarray(images, dtype = np.float32))
        np.save(dataset + '_labels', np.asarray(labels, dtype = np.int32))

    return

def load_data():
    X_train = np.load(PATH_TRAIN + '.npy') / 255
    y_train = np.load(PATH_TRAIN + '_labels.npy') / 255
    X_test = np.load(PATH_TEST + '.npy') / 255
    y_test = np.load(PATH_TEST + '_labels.npy') / 255
    return X_train, y_train, X_test, y_test

def main():
    get_data()
    return

if __name__ == "__main__":
    main()
