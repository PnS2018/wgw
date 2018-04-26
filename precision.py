import numpy as np
from keras.models import load_model

from DataProcessor import DataProcessor

# parameters
version = 6
num_classes = 2
image_size = 64
train_waldo = 'data/{}/train_waldo'.format(image_size)
train_notwaldo = 'data/{}/train_notwaldo'.format(image_size)
test_waldo = 'data/{}/test_waldo'.format(image_size)
test_notwaldo = 'data/{}/test_notwaldo'.format(image_size)

dp = DataProcessor(num_classes, image_size, train_waldo, train_notwaldo, test_waldo, test_notwaldo)

# load & preprocess data
train_x, train_y, test_x, test_y = dp.load_grayscale()
train_X, train_Y, test_X, test_Y, datagen = dp.preprocess_data(train_x, train_y, test_x, test_y)

model = load_model('version_{}.h5'.format(version))

preds_org = model.predict(test_x)
preds = np.round(preds_org)
false_pos = 0
false_neg = 0
total_pos = 0
total_neg = 0
for i in range(0, len(preds)):
    if test_Y[i][1] == 1:
        total_neg += 1
    if test_Y[i][0] == 1:
        total_pos += 1
    if preds[i][0] != test_Y[i][0] or preds[i][1] != test_Y[i][1]:
        if test_Y[i][1] == 1:
            false_pos += 1
        if test_Y[i][0] == 1:
            false_neg += 1

for i in range(0, 5):
    print(preds_org[i])
print('Total positives: {}, Total negatives: {}, False negatives: {}, False positives: {}'.format(total_pos, total_neg,
                                                                                                  false_neg, false_pos))
