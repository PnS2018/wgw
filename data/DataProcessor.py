import cv2
import os
import numpy as np

IMAGE_SIZE = 64


class DataProcessor:

    def __init__(self):
        self.image_size = 64
        self.train_waldo = '{}/train_waldo'.format(self.image_size)
        self.train_notwaldo = '{}/train_notwaldo'.format(self.image_size)
        self.test_waldo = '{}/test_waldo'.format(self.image_size)
        self.test_notwaldo = '{}/test_notwaldo'.format(self.image_size)

    def load_train_grayscale(self):
        train_x = []
        for image in os.listdir(self.train_waldo):
            train_x.append(cv2.imread('{}/{}'.format(self.train_waldo, image), 0))

        train_x = np.concatenate([image[np.newaxis] for image in train_x])
        train_y = []
        return train_x, train_y

    def load_test_grayscale(self):
        test_x = []
        test_y = []
        return test_x, test_y


if __name__ == '__main__':
    data_processor = DataProcessor()
    data_processor.load_train_grayscale()
