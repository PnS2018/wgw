import cv2
import numpy as np
from config import config


class RunProcessor:
    """
    Wrapper for functions that are for using the already trained network
    """

    def __init__(self):
        self.image_size = config.IMAGE_SIZE

    def get_wrapped_image(self, path):
        img = cv2.imread(path, 0)  # 0 for grayscale
        if not img.any():
            print ("No image found!")

        # amount of pixels added to the bottom so its divideable through size
        bottom = self.image_size - (img.shape[0] % self.image_size)
        # amount of pixels added to the right so its divideable through size
        right = self.image_size - (img.shape[1] % self.image_size)

        return cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=0)


    def get_image_tiles(self, wrap, stride):
        # calculate the amount of rows by dividing by stride
        # its sizeheight - size because otherwise it would produce another row at the end even when theres not enough pixel left for a tile
        rows = (wrap.shape[0] - self.image_size) // stride
        # calculate the amount of cols by dividing by stride (eg. 10 tiles possible cols=9 )
        # the size-1 is because the first pixel of the last possible tile must be considered while separating
        cols = (wrap.shape[1] - self.image_size) // stride

        img_tiles = []
        for row in range(0, rows):
            for col in range(0, cols):
                y1 = row * stride
                y2 = y1 + self.image_size - 1  # the 1 is because the start pixel is also part of the extracted picture
                x1 = col * stride
                x2 = x1 + self.image_size - 1

                print 'fetch tile. Row: %d, col: %d. Pixels: x: %d-%d, y: %d-%d' % (row, col, x1, x2, y1, y2)
                print 'tile array: %d' % (row * cols + col)

                img_tiles.append(wrap[y1:y2, x1:x2])

        return img_tiles

    def findWaldo(self, path, stride, model):
        img=self.get_wrapped_image(path)
        x= self.get_image_tiles(img, stride)
        rows = (img.shape[0] - self.image_size) // stride
        found=0
        pred=model.predict(x)

        for tile in pred.iterkeys(): # go through all predictions...
            if(pred[tile][0]): # ...and get the one, that are labeled true
                found+=1
                row=tile//rows
                col=tile-row*rows
                y1 = row * stride
                y2 = y1 + self.image_size - 1  # the 1 is because the start pixel is also part of the extracted picture
                x1 = col * stride
                x2 = x1 + self.image_size - 1
                img=cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        cv2.imshow('Found Waldos: '+found, img)

# img = cv2.imread('original-images/1.jpg')
# #img = cv2.imread('data/64/train_waldo/1_4_6.jpg')
# size=64
# stride=30
#
#
#
# img=cv2.rectangle(img, (100, 100), (164, 164), (255, 0, 0), 2)
# img=cv2.rectangle(img, (120, 100), (184, 164), (255, 0, 0), 3)
# img=cv2.rectangle(img, (140, 100), (204, 164), (255, 0, 0), 5)
# cv2.imshow('image1', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# rp= RunProcessor()
#
# img=rp.get_wrapped_image('original-images/1.jpg')
# cv2.imshow('image1', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()