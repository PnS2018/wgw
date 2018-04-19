from DataProcessor import DataProcessor
from ModelV1 import ModelManager

# parameters
version = 1
num_classes = 2
num_epochs = 10
image_size = 64
num_channels = 1
train_waldo = '{}/train_waldo'.format(image_size)
train_notwaldo = '{}/train_notwaldo'.format(image_size)
test_waldo = '{}/test_waldo'.format(image_size)
test_notwaldo = '{}/test_notwaldo'.format(image_size)

dp = DataProcessor(num_classes, image_size, train_waldo, train_notwaldo, test_waldo, test_notwaldo)

# load data
train_x, train_y, test_x, test_y = dp.load_grayscale()

print("[MESSAGE] Dataset is loaded.")

train_X, train_Y, test_X, test_Y, train_datagen, test_datagen = dp.preprocess_data(train_x, train_y, test_x, test_y)

print("[MESSAGE] Dataset is preprocessed")

model = ModelManager.get_model(image_size, image_size, num_channels)

print("[MESSAGE] Model is defined and compiled.")

# fits the model on batches with real-time data augmentation:
model.fit_generator(train_datagen.flow(train_x, train_Y, batch_size=64),
                    steps_per_epoch=len(train_x) / 64,
                    epochs=num_epochs,
                    callbacks=[])

print("[MESSAGE] Model is trained.")

# save the trained model
model.save("version_{}.h5".format(version))

print("[MESSAGE] Model is saved.")
