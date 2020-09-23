# imports from segmentation_models
from segmentation_models import Unet, Linknet, FPN, PSPNet
from segmentation_models.losses import jaccard_loss, categorical_crossentropy
from segmentation_models.metrics import iou_score, precision

# imports from keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

# imports from other project files
from .metrics import flat_iou, flat_overall_precision, flat_per_class_precision, overall_precision
from .custom_unet import custom_unet

# Set seeds
import tensorflow as tf
from numpy.random import seed
seed(2424)
tf.random.set_seed(2424)

BASE_MODELS = {'c-unet' : custom_unet,
               'unet' : Unet,
               'linknet': Linknet,
               'fpn' : FPN,
               'pspnet' : PSPNet,
               }

LOSSES = {'categorical_crossentropy' : 'categorical_crossentropy',
          'jaccard_loss' : jaccard_loss, }

def get_callbacks(early_stopping_patience):
    callbacks = []
    callbacks += [ModelCheckpoint('model.hdf5', monitor='val_loss',verbose=1, save_best_only=True)]
    callbacks += [EarlyStopping(monitor='val_loss', verbose=1, patience=early_stopping_patience)]
    return callbacks

def train_model(config, x_train, y_train, x_val, y_val):
    """
    config: {
        'model_type': 'unet',
        'backbone': 'mobilenet',
        'pretrained': False, # or 'imagenet',
        'loss': 'categorical_crossentropy',
        'learning_rate': 1e-4,
        'early_stopping_patience': 22,
        'batch_size': 32,
        'epochs': 99
    }
    """
    # Compile model
    NUM_CLASSES = config.get('num_classes', 3)
    IMSIZE = config.get('imshape', 256)

    model_type = config.get('model_type', 'unet')
    backbone = config.get('backbone', 'mobilenet')
    pretrained_weights = config.get('pretrained', None)
    learning_rate = config.get('learning_rate', 1e-4)

    loss = LOSSES[config.get('loss', 'categorical_crossentropy')]
    base_model = BASE_MODELS[model_type]

    if model_type == 'c-unet':
        model = base_model(input_size = (IMSIZE,IMSIZE, 1), classes = NUM_CLASSES)
    else:
        base_model = base_model(backbone, classes=NUM_CLASSES, activation='softmax', encoder_weights=pretrained_weights)
        # see https://segmentation-models.readthedocs.io/en/latest/tutorial.html#training-with-non-rgb-data
        # for training with non RGB data
        inp = Input(shape=(IMSIZE, IMSIZE, 1))
        l1 = Conv2D(3, (1, 1))(inp) # map 1 channels data to 3 channels
        out = base_model(l1)
        model = Model(inp, out, name=base_model.name)

    model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=[iou_score, precision, overall_precision])

    # Train model
    early_stopping_patience = config.get('early_stopping_patience', 22)
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 99)
    callbacks = get_callbacks(early_stopping_patience)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(x_val,y_val), verbose=2)
    model = load_model('model.hdf5', custom_objects={ 'iou_score': iou_score, 'precision': precision, 'overall_precision': overall_precision })
    return model

def score_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    iou = flat_iou(y_test, y_pred)
    overall_precision = flat_overall_precision(y_test, y_pred)
    per_class_precision = flat_per_class_precision(y_test, y_pred)

    return {
        'iou': iou,
        'overall_precision': overall_precision,
        'per_class_precision': per_class_precision
    }