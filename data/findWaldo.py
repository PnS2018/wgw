import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import Model



def getImageTiles(img, size, stride):  #
    if not img.any():
        print "No image found!"
        return -1

    bottom = size - (img.shape[0] % size)  # amount of pixels added to the bottom so its dividable through size
    right = size - (img.shape[1] % size)  # amount of pixels added to the right so its dividable through size
    img = cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=0)



img = cv2.imread('original-images/1.jpg')
#img = cv2.imread('data/64/train_waldo/1_4_6.jpg')
size=64
stride=64

if not img.any():
    print "No image found!"


bottom= size-(img.shape[0] % size) # amount of pixels added to the bottom so its divideable through size
right= size-(img.shape[1] % size) # amount of pixels added to the right so its divideable through size

wrap = cv2.copyMakeBorder(img,0,bottom,0,right,cv2.BORDER_CONSTANT,value=0)







rows = (img.shape[0]-size) // stride # calculate the amount of rows by dividing by stride (eg. 10 tiles possible rows=9 )
# its sizeheight - size because otherwise it would produce another row at the end even when theres not enough pixel left for a tile
cols = (img.shape[1]-size) // stride # calculate the amount of cols by dividing by stride (eg. 10 tiles possible cols=9 )

imgTiles=[]

for row in range(0, rows):
    for col in range(0, cols):
        y1= row*stride+1 # the +1 is because the numpy matrix array starts at 1
        y2=y1+size+1
        x1=col*stride+1
        x2=x1+size+1

        print 'fetch tile. Row: %d, col: %d. Pixels: x: %d-%d, y: %d-%d' %(row, col, x1, x2, y1, y2)
        print 'tile array: %d' %(row*(cols+1)+col)

        imgTiles[row*(cols+1)+col] = wrap[y1:y2, x1:x2]


#cv2.imshow('image',wrap)
cv2.imshow('image', imgTiles[0])
cv2.waitKey(0)
cv2.destroyAllWindows()