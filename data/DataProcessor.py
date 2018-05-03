import os

import cv2
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from config import config


class DataProcessor:
    def __init__(self):
        self.num_classes = config.NUM_CLASSES
        self.image_size = config.IMAGE_SIZE
        self.train_waldo = config.TRAIN_WALDO
        self.train_notwaldo = config.TRAIN_NOTWALDO
        self.test_waldo = config.TEST_WALDO
        self.test_notwaldo = config.TEST_NOTWALDO
        self.augmented_train_waldo = config.AUGMENTED_TRAIN_WALDO

    def load_grayscale(self):
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
        # augment training dataset
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(train_x)

        train_X = train_x
        test_X = datagen.standardize(test_x)

        # converting the input class labels to categorical labels for training
        train_Y = to_categorical(train_y, num_classes=self.num_classes)
        test_Y = to_categorical(test_y, num_classes=self.num_classes)

        return train_X, train_Y, test_X, test_Y, datagen

    def augment_and_save(self):
        for image in os.listdir(self.train_waldo):
            img = load_img('{}/{}'.format(self.train_waldo, image))  # this is a PIL image
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

            datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
                # zca_whitening=True, #changes structure and outline of pictures
            )

            i = 0
            for _ in datagen.flow(x,
                                  batch_size=1,
                                  save_to_dir=self.augmented_train_waldo,
                                  save_prefix='augmented',
                                  save_format='jpeg'):
                i += 1
                if i > 3:
                    break  # otherwise the generator would loop indefinitely

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
