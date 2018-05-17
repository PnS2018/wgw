import cv2
import numpy as np

from utils.config import config


class RunProcessor:
    """
    Wrapper for functions that are for using the already trained network
    """

    def __init__(self):
        self.image_size = config.IMAGE_SIZE

    def get_wrapped_image(self, path):
        img = cv2.imread(path, 0)  # 0 for grayscale

        # amount of pixels added to the bottom so its divideable through size
        bottom = self.image_size - (img.shape[0] % self.image_size)
        # amount of pixels added to the right so its divideable through size
        right = self.image_size - (img.shape[1] % self.image_size)

        return cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=0)

    def get_image_tiles(self, wrap, stride):
        # calculate the amount of rows by dividing by stride
        rows = (wrap.shape[0] - self.image_size) // stride
        # calculate the amount of cols by dividing by stride (eg. 10 tiles possible cols=9 )
        cols = (wrap.shape[1] - self.image_size) // stride

        img_tiles = []
        for row in range(0, rows+1):
            for col in range(0, cols+1):
                y1 = row * stride
                y2 = y1 + self.image_size
                x1 = col * stride
                x2 = x1 + self.image_size

                print('fetch tile. Row: %d, col: %d. Pixels: x: %d-%d, y: %d-%d' % (row, col, x1, x2, y1, y2))
                print('tile array: %d' % len(img_tiles))

                img_tiles.append(wrap[y1:y2, x1:x2])

        return img_tiles

    def find_waldo(self, path, stride, model):
        img = self.get_wrapped_image(path)
        x = self.get_image_tiles(img, stride)

        # bring them into the right format for use in the model
        x = np.concatenate([image[np.newaxis, :] for image in x])  # concatenate the images along the first axis
        x = x[:, :, :, np.newaxis]  # add a forth axis

        rows = (img.shape[0] - self.image_size) // stride
        cols = (img.shape[1] - self.image_size) // stride
        found = 0
        preds_org = model.predict(x)
        preds = np.round(preds_org)

        for i in range(0, len(preds)):
            if preds[i][0]:  # ...and get the one, that are labeled true
                found += 1
                row = i // cols - 1
                col = i - row * (cols + 1)
                y1 = row * stride
                y2 = y1 + self.image_size
                x1 = col * stride
                x2 = x1 + self.image_size
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                print('fetch tile. Row: %d, col: %d. Pixels: x: %d-%d, y: %d-%d' % (row, col, x1, x2, y1, y2))

        cv2.imshow('Found Waldos: {}'.format(found), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
