import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import Model



def getImageTiles(img, size, stride):  #
    if not img.any():
        print "No image found!"

    bottom = size - (img.shape[0] % size)  # amount of pixels added to the bottom so its divideable through size
    right = size - (img.shape[1] % size)  # amount of pixels added to the right so its divideable through size
    wrap = cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=0)

    rows = (wrap.shape[0] - size) // stride  # calculate the amount of rows by dividing by stride
    # its sizeheight - size because otherwise it would produce another row at the end even when theres not enough pixel left for a tile
    cols = (wrap.shape[
                1] - size) // stride  # calculate the amount of cols by dividing by stride (eg. 10 tiles possible cols=9 )
    # the size-1 is because the first pixel of the last possible tile must be considered while separating
    imgTiles = []

    # print 'There are %d rows and %d tiles' %(rows, cols)
    for row in range(0, rows):
        for col in range(0, cols):
            y1 = row * stride
            y2 = y1 + size - 1  # the 1 is because the start pixel is also part of the extracted picture
            x1 = col * stride
            x2 = x1 + size - 1

            print 'fetch tile. Row: %d, col: %d. Pixels: x: %d-%d, y: %d-%d' % (row, col, x1, x2, y1, y2)
            print 'tile array: %d' % (row * cols + col)

            # imgTiles[row*(cols+1)+col] = wrap[y1:y2, x1:x2]
            imgTiles.append(wrap[y1:y2, x1:x2])

    return imgTiles

# img = cv2.imread('original-images/1.jpg')
# #img = cv2.imread('data/64/train_waldo/1_4_6.jpg')
# size=64
# stride=30
# 
#
# print wrap.shape
# print len(imgTiles)
# #cv2.imshow('image',wrap)
# cv2.imshow('image1', imgTiles[2717])
# cv2.imshow('image2', imgTiles[2718])
# cv2.imshow('image3', imgTiles[2719])
# cv2.waitKey(0)
# cv2.destroyAllWindows()