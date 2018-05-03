from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model

from config import config


class ModelManager(object):

    @staticmethod
    def get_model():
        x = Input((config.IMAGE_SIZE, config.IMAGE_SIZE, config.NUMBER_OF_CHANNELS))
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
        d = Dense(200, activation='relu')(f)
        y = Dense(2, activation='softmax')(d)
        model = Model(x, y)

        # print model summary
        model.summary()

        # compile the model against the categorical cross entropy loss
        # loss:       cost function, should be cross validation, categorical_crossentropy also possible
        # optimizer:  schedule learning rate troughout the training
        # metric:    performance measure categroical_crossentropy, AUC for binaryclassf, or accuracy
        model.compile(loss='binary_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
        return model
