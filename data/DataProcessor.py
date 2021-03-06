import os

import cv2
import numpy as np
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from keras.utils import to_categorical

from utils.config import config


class DataProcessor:
    """
    Wrapper used for all data manipulation actions
    """

    def __init__(self):
        self.image_size = config.IMAGE_SIZE
        self.train_waldo = config.TRAIN_WALDO
        self.train_notwaldo = config.TRAIN_NOTWALDO
        self.test_waldo = config.TEST_WALDO
        self.test_notwaldo = config.TEST_NOTWALDO
        self.augmented_train_waldo = config.AUGMENTED_TRAIN_WALDO

    def load_grayscale(self):
        """
        Load the train and test images as grayscale vectors from the data directory
        :return: numpy arrays of the respective train and test images
        """

        # a truth value of 0 corresponds to waldo and 1 to notwaldo
        train_waldo_x, train_waldo_y = self._load_from_path_grayscale(self.train_waldo, 0)
        train_notwaldo_x, train_notwaldo_y = self._load_from_path_grayscale(self.train_notwaldo, 1)
        test_waldo_x, test_waldo_y = self._load_from_path_grayscale(self.test_waldo, 0)
        test_notwaldo_x, test_notwaldo_y = self._load_from_path_grayscale(self.test_notwaldo, 1)

        train_x = np.concatenate((train_waldo_x, train_notwaldo_x))
        train_y = np.concatenate((train_waldo_y, train_notwaldo_y))
        test_x = np.concatenate((test_waldo_x, test_notwaldo_x))
        test_y = np.concatenate((test_waldo_y, test_notwaldo_y))

        train_x = train_x.astype("float32")
        test_x = test_x.astype("float32")

        return train_x, train_y, test_x, test_y

    def preprocess_data(self, train_x, train_y, test_x, test_y):
        """
        Preprocess and augment the data to be used in the model
        :return: numpy arrays with the augmented train and test vectors
        """
        # augment training dataset
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest')

        datagen.fit(train_x)

        # leave training data set as it is and standardize the test vector set
        train_X = train_x
        test_X = datagen.standardize(test_x)

        # converting the input class labels to categorical labels for training
        train_Y = to_categorical(train_y, num_classes=2)
        test_Y = to_categorical(test_y, num_classes=2)

        return train_X, train_Y, test_X, test_Y, datagen

    def augment_and_save(self):
        """
        An augmentation function to generate augmented RGB pictures that saves them back to disk
        """
        for image in os.listdir(self.train_waldo):
            img = load_img('{}/{}'.format(self.train_waldo, image))
            x = img_to_array(img)  # this is a numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a numpy array with shape (1, 3, 150, 150)

            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            i = 0
            for _ in datagen.flow(x,
                                  batch_size=1,
                                  save_to_dir=self.augmented_train_waldo,
                                  save_prefix='augmented',
                                  save_format='jpeg'):
                i += 1
                if i > 3:
                    break

    def _load_from_path_grayscale(self, path, label):
        x = []
        # load images from path
        for image in os.listdir(path):
            x.append(cv2.imread('{}/{}'.format(path, image), 0))

        # bring them into the right format for use in the model
        x = np.concatenate([image[np.newaxis, :] for image in x])  # concatenate the images along the first axis
        x = x[:, :, :, np.newaxis]  # add a forth axis

        # add truth values
        if label == 1:
            y = np.ones(x.shape[0])
        else:
            y = np.zeros(x.shape[0])

        return x, y
