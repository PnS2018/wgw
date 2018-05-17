from keras.models import load_model

from data.RunProcessor import RunProcessor
from utils.config import config

run_processor = RunProcessor()
model = load_model('version{}_epochs{}_modelv4.h5'.format(config.VERSION, config.NUM_EPOCHS))
model.summary()

run_processor.find_waldo(config.RUN_IMAGE_PATH, config.STRIDE, model)
