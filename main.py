import numpy as np
import matplotlib.pyplot as plt

from DataProcessor import DataProcessor
from ModelV1 import ModelManager

# parameters
NUM_CLASSES = 2
NUM_EPOCHS = 10

# load data
train_x, train_y, test_x, test_y = DataProcessor.load_grayscale()

print("[MESSAGE] Dataset is loaded.")

train_X, train_Y, test_X, test_Y = DataProcessor.preprocess_data(train_x, train_y, test_x, test_y, NUM_CLASSES)

print("[MESSAGE] Dataset is preprocessed")

model = ModelManager.get_model(train_X.shape[1], train_X.shape[2], train_X.shape[3])

print("[MESSAGE] Model is defined and compiled.")

if mode == 'Train':
    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(train_x, train_Y, batch_size=64),
                        steps_per_epoch=len(train_x) / 64, epochs=NUM_EPOCHS,
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
