from config import config
from DataProcessor import DataProcessor
from ModelV3 import ModelManager

dp = DataProcessor()

# load data
train_x, train_y, test_x, test_y = dp.load_grayscale()

print("[MESSAGE] Dataset is loaded.")

train_X, train_Y, test_X, test_Y, datagen = dp.preprocess_data(train_x, train_y, test_x, test_y)

print("[MESSAGE] Dataset is preprocessed")

model = ModelManager.get_model()

print("[MESSAGE] Model is defined and compiled.")

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(train_x, train_Y, batch_size=64),
                    epochs=config.NUM_EPOCHS,
                    callbacks=[],
                    validation_data=(test_X, test_Y))

print("[MESSAGE] Model is trained.")

# save the trained model
model.save("version{}_epochs{}.h5".format(config.VERSION, config.NUM_EPOCHS))

print("[MESSAGE] Model is saved.")
