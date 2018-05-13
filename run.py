from keras.models import load_model

from data.RunProcessor import RunProcessor
from utils.config import config

run_processor = RunProcessor()
model = load_model('version{}_epochs{}.h5'.format(config.VERSION, config.NUM_EPOCHS))

run_processor.find_waldo(config.RUN_IMAGE_PATH, config.STRIDE, model)
