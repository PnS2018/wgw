import os

import cv2
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


class DataProcessor:
    def __init__(self, num_classes, image_size, train_waldo, train_notwaldo, test_waldo, test_notwaldo):
        self.num_classes = num_classes
        self.image_size = image_size
        self.train_waldo = train_waldo
        self.train_notwaldo = train_notwaldo
        self.test_waldo = test_waldo
        self.test_notwaldo = test_notwaldo

    def load_grayscale(self):
        train_waldo_x, train_waldo_y = self._load_from_path_grayscale(self.train_waldo, 0)
        train_notwaldo_x, train_notwaldo_y = self._load_from_path_grayscale(self.train_notwaldo, 1)
        test_waldo_x, test_waldo_y = self._load_from_path_grayscale(self.test_waldo, 0)
        test_notwaldo_x, test_notwaldo_y = self._load_from_path_grayscale(self.test_notwaldo, 1)

        train_x = np.concatenate((train_waldo_x, train_notwaldo_x))
        train_y = np.concatenate((train_waldo_y, train_notwaldo_y))
        test_x = np.concatenate((test_waldo_x, test_notwaldo_x))
        test_y = np.concatenate((test_waldo_y, test_notwaldo_y))

        return train_x, train_y, test_x, test_y

    def preprocess_data(self, train_x, train_y, test_x, test_y):
        # augment training dataset
        train_datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator()

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        train_datagen.fit(train_x)

        train_X = train_x
        test_X = train_datagen.standardize(test_x)

        # converting the input class labels to categorical labels for training
        train_Y = to_categorical(train_y, num_classes=self.num_classes)
        test_Y = to_categorical(test_y, num_classes=self.num_classes)

        return train_X, train_Y, test_X, test_Y, train_datagen, test_datagen

    def _load_from_path_grayscale(self, path, label):
        x = []
        # load images from path
        for image in os.listdir(path):
            x.append(cv2.imread('{}/{}'.format(path, image), 0))

        # bring them into the right format for use in the model
        x = np.concatenate([image[np.newaxis, :] for image in x])  # concatenate the images along the first axis
        x = x[:, :, :, np.newaxis]  # add a forth axis

        if label == 1:
            y = np.ones(x.shape[0])
        else:
            y = np.zeros(x.shape[0])

        return x, y
