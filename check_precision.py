import numpy as np
from keras.models import load_model

from data.DataProcessor import DataProcessor
from utils.config import config

dp = DataProcessor()

# load & preprocess data
train_x, train_y, test_x, test_y = dp.load_grayscale()
_, _, test_X, test_Y, datagen = dp.preprocess_data(train_x, train_y, test_x, test_y)

model = load_model('version{}_epochs{}_modelv4.h5'.format(config.VERSION, config.NUM_EPOCHS))

preds_org = model.predict(test_X)
preds = np.round(preds_org - config.THRESHOLD + 0.5)
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

print('Total positives: {}, Total negatives: {}, False negatives: {}, False positives: {}'.format(total_pos, total_neg,
                                                                                                  false_neg, false_pos))
