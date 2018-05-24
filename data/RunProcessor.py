import heapq
from datetime import datetime

import cv2
import numpy as np

from utils.config import config


class RunProcessor:
    """
    Wrapper for functions that are for using the already trained network
    """

    def __init__(self):
        self.image_size = config.IMAGE_SIZE
        self.mean = config.MEAN
        self.std = config.STD
        self.top_x = config.TOP_X
        self.threshold = config.THRESHOLD

    def find_waldo(self, path, stride, model):
        """
        Function that takes an image and displays the image with the prediction visualized in it
        :param path: path to image file
        :param stride: the stride that should be used for tiling the image
        :param model: model that should be used for our prediction
        """
        img = self._get_wrapped_image(path)
        x_org = self._get_image_tiles(img, stride)

        # allow drawing to be of color
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # bring them into the right format for use in the model
        x = np.concatenate([image[np.newaxis, :] for image in x_org])  # concatenate the images along the first axis
        x = x[:, :, :, np.newaxis]  # add a forth axis
        x = x.astype("float32")
        x = (x - self.mean) / self.std  # standardize

        cols = ((img.shape[1] - self.image_size) // stride) + 1
        found = 0

        start = datetime.now()
        preds = model.predict(x)
        end = datetime.now()
        print('The function took {} seconds to complete'.format(end - start))

        preds = [x[0] for x in preds]
        max_preds = heapq.nlargest(self.top_x, preds)  # find the top_x predicted elements
        max_pred = max(preds)  # find the best prediction

        for i in range(0, len(preds)):
            if preds[i] > self.threshold:
                found += 1
                if preds[i] in max_preds:
                    row = i // cols
                    col = i - row * cols
                    y1 = row * stride
                    y2 = y1 + self.image_size
                    x1 = col * stride
                    x2 = x1 + self.image_size
                    if preds[i] == max_pred:
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    else:
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    print('Pixels: x: %d-%d, y: %d-%d, Prediction: %f' % (x1, x2, y1, y2, preds[i]))

        cv2.imshow('Found in total {} Waldos above our threshold {}.'.format(found, self.threshold), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _get_wrapped_image(self, path):
        img = cv2.imread(path, 0)  # 0 for greyscale

        # amount of pixels added to the bottom so its dividable through size
        bottom = self.image_size - (img.shape[0] % self.image_size) % self.image_size
        # amount of pixels added to the right so its dividable through size
        right = self.image_size - (img.shape[1] % self.image_size) % self.image_size

        return cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_REFLECT)

    def _get_image_tiles(self, wrap, stride):
        # calculate the amount of rows by dividing by stride
        rows = ((wrap.shape[0] - self.image_size) // stride) + 1
        # calculate the amount of cols by dividing by stride
        cols = ((wrap.shape[1] - self.image_size) // stride) + 1

        img_tiles = []
        for row in range(0, rows):
            for col in range(0, cols):
                y1 = row * stride
                y2 = y1 + self.image_size
                x1 = col * stride
                x2 = x1 + self.image_size
                img_tiles.append(wrap[y1:y2, x1:x2])

        return img_tiles
