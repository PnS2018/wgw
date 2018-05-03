from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.losses import categorical_crossentropy
from keras.models import Model
import keras.backend as K



def recall(y_true, y_pred):
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


def precision(y_true, y_pred):
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


def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))


class ModelManager(object):


    @staticmethod
    def get_model(image_dim_0, image_dim_1, number_of_channels):
        x = Input((image_dim_0, image_dim_1, number_of_channels))
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
        model.compile(loss=categorical_crossentropy,
                      optimizer='sgd',
                      metrics=['accuracy', f1, precision, recall])
        return model
