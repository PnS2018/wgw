from RunProcessor import RunProcessor
from ModelManager import ModelManager

imgPath='data/original-images/1.jpg'
weightsPath = ''
stride= 32


rp = RunProcessor()
mm = ModelManager()
model = mm.get_model_v3()
model.load_weights(weightsPath)

rp.findWaldo(imgPath, stride, model)




# TODO
