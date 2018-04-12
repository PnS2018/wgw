import cv2
import os
import numpy as np

IMAGE_SIZE = 64
TRAIN_WALDO = '{}/train_waldo'.format(IMAGE_SIZE)
TRAIN_NOTWALDO = '{}/train_notwaldo'.format(IMAGE_SIZE)
TEST_WALDO = '{}/test_waldo'.format(IMAGE_SIZE)
TEST_NOTWALDO = '{}/test_notwaldo'.format(IMAGE_SIZE)


class DataProcessor:

    @staticmethod
    def load_train_grayscale():
        waldo_x, waldo_y = DataProcessor._load_from_path_grayscale(TRAIN_WALDO, 0)
        notwaldo_x, notwaldo_y = DataProcessor._load_from_path_grayscale(TRAIN_NOTWALDO, 1)
        return np.concatenate((waldo_x, notwaldo_x)), np.concatenate((waldo_y, notwaldo_y))

    @staticmethod
    def load_test_grayscale():
        waldo_x, waldo_y = DataProcessor._load_from_path_grayscale(TEST_WALDO, 0)
        notwaldo_x, notwaldo_y = DataProcessor._load_from_path_grayscale(TEST_NOTWALDO, 1)
        return np.concatenate((waldo_x, notwaldo_x)), np.concatenate((waldo_y, notwaldo_y))

    @staticmethod
    def _load_from_path_grayscale(path, label):
        x = []
        for image in os.listdir(path):
            x.append(cv2.imread('{}/{}'.format(path, image), 0))

        x = np.concatenate([image[np.newaxis, :] for image in x])
        x = x[:, :, :, np.newaxis]

        if label == 1:
            y = np.ones(x.shape[0])
        else:
            y = np.zeros(x.shape[0])

        return x, y


if __name__ == '__main__':
    train_x, train_y = DataProcessor.load_train_grayscale()
    test_x, test_y = DataProcessor.load_test_grayscale()
    print('')
