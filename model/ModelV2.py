from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.losses import categorical_crossentropy
from keras.models import Model
import keras.backend as K
import f1 as f1

from config import config


class ModelManager(object):


    @staticmethod
    def get_model():
        x = Input((config.IMAGE_SIZE, config.IMAGE_SIZE, config.NUM_CHANNELS))
        c1 = Conv2D(filters=20,
                    kernel_size=(7, 7),
                    padding='same',
                    activation='relu')(x)
        p1 = MaxPooling2D((2, 2))(c1)
        c2 = Conv2D(filters=25,
                    kernel_size=(5, 5),
                    padding='same',
                    activation='relu')(p1)
        p2 = MaxPooling2D((2, 2))(c2)
        f = Flatten()(p2)
        d = Dense(20, activation='relu')(f)
        y = Dense(2, activation='softmax')(d)
        model = Model(x, y)

        # print model summary
        model.summary()

        # compile the model against the categorical cross entropy loss
        # loss:       cost function, should be cross validation, categorical_crossentropy also possible
        # optimizer:  schedule learning rate troughout the training
        # metric:    performance measure categroical_crossentropy, AUC for binaryclassf, or accuracy
        model.compile(loss= test, # categorical_crossentropy
                      optimizer='sgd',
                      metrics=['accuracy', f1.f1, f1.precision, f1.recall])
        return model
