import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from DataProcessor import DataProcessor
from ModelV1 import Mode

# load data
train_x, train_y = DataProcessor.load_train_grayscale()
test_x, test_y = DataProcessor.load_test_grayscale()

num_classes = 2
num_epochs = 10
mode = 'Test'

print("[MESSAGE] Dataset is loaded.")

# preprocessing for training and testing images
train_x = train_x.astype("float32") / 255.  # rescale image
mean_train_x = np.mean(train_x, axis=0)  # compute the mean across pixels
train_X = train_x - mean_train_x  # remove the mean pixel value from image
test_x = test_x.astype("float32") / 255.
test_X = test_x - mean_train_x

print("[MESSAGE] Dataset is preprocessed.")

# converting the input class labels to categorical labels for training
train_Y = to_categorical(train_y, num_classes=num_classes)
test_Y = to_categorical(test_y, num_classes=num_classes)

print("[MESSAGE] Converted labels to categorical labels.")

model = ModelManager.get_model(train_x.shape[1], train_x.shape[2], train_x.shape[3])

print("[MESSAGE] Model is defined and compiled.")

# augment training dataset
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(train_x)

# augment testing dataset
test_x = test_x / (datagen.std + datagen.zca_epsilon)

if mode == 'Train':
    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(train_x, train_Y, batch_size=64),
                        steps_per_epoch=len(train_x) / 64, epochs=num_epochs,
                        callbacks=[])

    print("[MESSAGE] Model is trained.")

    # save the trained model
    model.save("conv-net-fashion-mnist-trained.hdf5")

    print("[MESSAGE] Model is saved.")

    # visualize the ground truth and prediction
    # take first 10 examples in the testing dataset
    test_x_vis = test_x[:10]  # fetch first 10 samples
    ground_truths = test_y[:10]  # fetch first 10 ground truth prediction
    # predict with the model
    preds = np.argmax(model.predict(test_x_vis), axis=1).astype(np.int)

    labels = ["Tshirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
              "Shirt", "Sneaker", "Bag", "Ankle Boot"]

    plt.figure()
    for i in range(2):
        for j in range(5):
            plt.subplot(2, 5, i * 5 + j + 1)
            plt.imshow(np.squeeze(test_x[i * 5 + j]), cmap="gray")
            plt.title("Ground Truth: %s, \n Prediction %s" %
                      (labels[ground_truths[i * 5 + j]],
                       labels[preds[i * 5 + j]]))
    plt.show()

if mode is 'Test':
    # load the trained model
    model.load_weights("conv-net-fashion-mnist-trained.hdf5")

    preds = np.argmax(model.predict(test_x), axis=1).astype(np.int)
    num_of_wrong_cat = np.sum(preds != test_y)
    perc_of_wrong_cat = (10000 - num_of_wrong_cat) / 100.0

    print('Number of wrong categorizations: {} and percent right: {}%'.format(num_of_wrong_cat, perc_of_wrong_cat))
