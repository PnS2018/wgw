from keras.models import load_model

from data.RunProcessor import RunProcessor
from utils.config import config

run_processor = RunProcessor()

print("Starting prediction run.")

model = load_model('version{}_epochs{}_model{}.h5'.format(config.VERSION, config.NUM_EPOCHS, config.MODEL_VERSION))
model.summary()

run_processor.find_waldo(config.RUN_IMAGE_PATH, config.STRIDE, model)

print("Finishing prediction run.")
