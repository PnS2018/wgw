from data.DataProcessor import DataProcessor
from utils.config import config

dp = DataProcessor()

print("Starting augmentation.")

dp.augment_and_save()

print("Finished augmentation. The images were saved to {}".format(config.AUGMENTED_TRAIN_WALDO))
