import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from imblearn.metrics import geometric_mean_score
from keras_vggface.vggface import VGGFace
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Dense, Dropout, Activation, Input, BatchNormalization
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.models import load_model, Sequential, Model
from keras.regularizers import l2
from keras.layers.merge import concatenate
from scipy.stats import pearsonr
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, make_scorer, r2_score
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, maxabs_scale
from keras.optimizers import Adam, SGD
import keras.backend as K


def build_model(neurons=[32, 8],
                droprate=0.5,
                activation=LeakyReLU(),
                optimizer=Adam(lr=0.0001),
                feature_model_type='resnet50',
                n_hidden_layers=2,
                n_nodes_output_cat=3,
                bmomentum=0.99,
                bepsilon=0.001,
                reg_val=0.001):

    # ------------------------------------
    # The first part of the model consists
    # of the pre-trained VGG face model
    # ------------------------------------

    # Grab relavent part of network
    if feature_model_type == 'vgg16':
        vgg_model = VGGFace(model='vgg16')
        n_lyrs = 22
        for l in range(4):
            vgg_model.layers.pop()
            vgg_model.outputs = [vgg_model.layers[-1].output]
            vgg_model.layers[-1].outbound_nodes = []

    elif feature_model_type == 'resnet50':
        vgg_model = VGGFace(model='resnet50')
        out = vgg_model.get_layer('flatten_1').output
        n_lyrs = 175
        vgg_model.layers.pop()
        vgg_model.outputs = [vgg_model.layers[-1].output]
        vgg_model.layers[-1].outbound_nodes = []

    # Add input layer for age and gender
    age_gender_input = Input(shape=(2,), name='age_gender_input')

    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.outputs)

    # Merge auxilary input to end of pretrained model
    x = concatenate([vgg_model.output, age_gender_input])

    # Add Batch Normalization. Auxillary inputs are on a different scale as VGG outputs
    x = BatchNormalization(momentum=bmomentum, epsilon=bepsilon)(x)

    # First new layer
    if reg_val is None:
        x_cat = Dense(neurons[0])(x)
        x_cont = Dense(neurons[0])(x_cont)
    else:
        x_cat = Dense(neurons[0], kernel_regularizer=l2(reg_val))(x)
        x_cont = Dense(neurons[0], kernel_regularizer=l2(reg_val))(x)

    x_cat = activation(x_cat)
    x_cat = Dropout(droprate)(x_cat)
    x_cont = activation(x_cont)
    x_cont = Dropout(droprate)(x_cont)

    # Create split layers for classification and for regression
    for n in range(1, n_hidden_layers):
        if reg_val is None:
            x_cat = Dense(neurons[n])(x_cat)
            x_cont = Dense(neurons[n])(x_cont)
        else:
            x_cat = Dense(neurons[n], kernel_regularizer=l2(reg_val))(x_cat)
            x_cont = Dense(neurons[n], kernel_regularizer=l2(reg_val))(x_cont)

        x_cat = activation(x_cat)
        x_cat = Dropout(droprate)(x_cat)
        x_cont = activation(x_cont)
        x_cont = Dropout(droprate)(x_cont)

    # Add output layers for classification and regression tasks
    x_cat = Dense(n_nodes_output_cat, activation='softmax', name='cat_out')(x_cat)
    x_cont = Dense(1, activation='linear', name='cont_out')(x_cont)

    model = Model(
        inputs=[vgg_model.input, age_gender_input],
        outputs=[x_cat, x_cont]
    )

    # Freeze layers associated with VGGFace model
    for layer in model.layers[:n_lyrs]:
        layer.trainable = False

    model.compile(
        loss={'cat_out': 'categorical_crossentropy', 'cont_out': 'mean_squared_error'},
        optimizer=optimizer,
        metrics={'cat_out': ['acc'], 'cont_out': ['mse']}
    )

    return model


if __name__ == '__main__':

    test_model = build_model()
