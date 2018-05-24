import numpy as np

from data.DataProcessor import DataProcessor
from model.ModelManager import ModelManager
from utils.config import config

np.random.seed(config.VERSION)
dp = DataProcessor()

print("Starting model training.")

# load data
train_x, train_y, test_x, test_y = dp.load_grayscale()
train_X, train_Y, test_X, test_Y, datagen = dp.preprocess_data(train_x, train_y, test_x, test_y)
model = ModelManager().get_model_v3()

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(train_x, train_Y, batch_size=64),
                    epochs=config.NUM_EPOCHS,
                    callbacks=[],
                    validation_data=(test_X, test_Y))

# save the trained model
model.save("version{}_epochs{}_model{}.h5".format(config.VERSION, config.NUM_EPOCHS, config.MODEL_VERSION))

print("Finishing model training.")
