from keras.models import load_model

from RunProcessor import RunProcessor

version = 1
image_size = 64

rp = RunProcessor(image_size)

blocks = rp.get_image_tiles('data/original-images/1.jpg', 64)

model = load_model('version_{}.h5'.format(version))
