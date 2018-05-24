import keras.backend as K
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.models import Model

from utils.config import config


class ModelManager:
    """
    Wrapper for our models
    """

    def __init__(self):
        self.image_x_axis = config.IMAGE_SIZE
        self.image_y_axis = config.IMAGE_SIZE

    def get_model_v1(self):
        """
        Our model MVP
        :return: model instance
        """
        x = Input((self.image_x_axis, self.image_y_axis, 1))
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
        # metric:     performance measure categroical_crossentropy, AUC for binaryclassf, or accuracy
        model.compile(loss='binary_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
        return model

    def get_model_v2(self):
        """
        Model that uses f1
        :return: model instance
        """
        x = Input((self.image_x_axis, self.image_y_axis, 1))
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
        # metric:     performance measure categroical_crossentropy, AUC for binaryclassf, or accuracy
        model.compile(loss='binary_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy', _f1, _precision, _recall])
        return model

    def get_model_v3(self):
        """
        A bit more refined model
        :return: model instance
        """
        x = Input((self.image_x_axis, self.image_y_axis, 1))
        c1 = Conv2D(filters=20,
                    kernel_size=(7, 7),
                    strides=(2, 2),
                    padding='same',
                    activation='relu')(x)
        p1 = MaxPooling2D((2, 2))(c1)
        c2 = Conv2D(filters=25,
                    kernel_size=(7, 7),
                    strides=(2, 2),
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
        # metric:     performance measure categroical_crossentropy, AUC for binaryclassf, or accuracy
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def get_model_v4(self):
        """
        A bit more refined model
        :return: model instance
        """
        x = Input((self.image_x_axis, self.image_y_axis, 1))
        c1 = Conv2D(filters=20,
                    kernel_size=(7, 7),
                    strides=(2, 2),
                    padding='same',
                    activation='relu')(x)
        p1 = MaxPooling2D((2, 2))(c1)
        c2 = Conv2D(filters=25,
                    kernel_size=(7, 7),
                    strides=(2, 2),
                    padding='same',
                    activation='relu')(p1)
        p2 = MaxPooling2D((2, 2))(c2)
        c3 = Conv2D(filters=15,
                    kernel_size=(7, 7),
                    strides=(2, 2),
                    padding='same',
                    activation='relu')(p2)
        p3 = MaxPooling2D((2, 2))(c3)
        f = Flatten()(p3)
        d = Dense(20, activation='relu')(f)
        y = Dense(2, activation='softmax')(d)
        model = Model(x, y)

        # print model summary
        model.summary()

        # compile the model against the categorical cross entropy loss
        # loss:       cost function, should be cross validation, categorical_crossentropy also possible
        # optimizer:  schedule learning rate troughout the training
        # metric:     performance measure categroical_crossentropy, AUC for binaryclassf, or accuracy
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


def _recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    y_true = K.cast(K.argmax(y_true, axis=1), 'float32')
    y_pred = K.cast(K.argmax(y_pred, axis=1), 'float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def _precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_true = K.cast(K.argmax(y_true, axis=1), 'float32')
    y_pred = K.cast(K.argmax(y_pred, axis=1), 'float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def _f1(y_true, y_pred):
    prec = _precision(y_true, y_pred)
    rec = _recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))
