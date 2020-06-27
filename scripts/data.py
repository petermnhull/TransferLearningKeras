import os
import numpy as np
from skimage.io import imread
from pathlib import Path
import cv2
from myconfig import CLASS_NAMES, IMAGE_SHAPE, PATH_TRAIN, PATH_TEST, PATH_DATA, SHUFFLE
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Directory
dir = os.path.dirname(__file__)

def get_data():
    for dataset in [PATH_TRAIN, PATH_TEST]:
        # Initialise
        images = []
        labels = []
        for i in range(len(CLASS_NAMES)):
            # Get paths
            path = os.path.join(dir, PATH_DATA, dataset, CLASS_NAMES[i])
            paths = Path(path).glob('**/*.jpg')
            for path in paths:
                # Process image
                image = imread(path)
                image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]), IMAGE_SHAPE[2])
                if image.shape == (IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]):
                    images.append(image)
                    labels.append(i)
                print('Done: ', path)

        if SHUFFLE:
            images, labels = shuffle(images, labels)

        # Export arrays to file
        np.save(dataset, np.asarray(images, dtype = np.float32))
        np.save(dataset + '_labels', np.asarray(labels, dtype = np.int32))

    return

def load_data():
    X_train = np.load(PATH_TRAIN + '.npy') / 255
    y_train = np.load(PATH_TRAIN + '_labels.npy')
    X_test = np.load(PATH_TEST + '.npy') / 255
    y_test = np.load(PATH_TEST + '_labels.npy')
    return X_train, y_train, X_test, y_test

def shuffle(images, labels):
    assert len(images) == len(labels)
    p = np.random.permutation(len(images))
    return images[p], labels[p]

def main():
    get_data()
    return

if __name__ == "__main__":
    main()
