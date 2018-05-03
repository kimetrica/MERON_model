import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from imblearn.metrics import geometric_mean_score
from keras_vggface.vggface import VGGFace
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout, Activation
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.models import load_model, Sequential, Model
from keras.regularizers import l2
from scipy.stats import pearsonr
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, make_scorer, r2_score
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, maxabs_scale


def plot_confusion_matrix(true_vals, predicted_vals, classes,
                          normalize=True,
                          savefig=False,
                          title='Confusion matrix for BMI predicitions',
                          cmap=plt.cm.YlOrRd):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.


    Parameters
    ----------
    true_vals : Numpy array, shape = [n_samples]
                Actual values to compare against prediction
    predicted_vals : Numpy array, shape = [n_samples]
                     Model predicted values
    classes : Array, shape = [n_samples]
              List of labels to index the matrix. This may be used to reoder or select a subset of
              labels. If none is given, those that appear at least once in true_vals or predicted_vals
              are used in sorted order.
    savefig : Logical or string
              If False no image of confusion matrix is saved else the fname of the image for the
              confusion matrix.
    title : String
            Title for confusion matrix plot
    cmap : Matplotlib colormap object
           Colormap for confusion matrix
    """
    cm = confusion_matrix(true_vals, predicted_vals, labels=classes)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    if savefig:
        plt.savefig(savefig, dpi=350)


class Meron(object):
    '''A Convolutional Neural Network for the detection of malnutrition using facial imagery

    Parameters
    ----------
    pred_type : string
                Type of malnutrition prediction to perform -
                    'classification' -- Classify the malnutrition status of an individual image
                    'regression' -- Prediction of the Weight for Height Z-score (WHZ)
    '''

    def __init__(self, pred_type='classification'):

        self.pred_type = pred_type

        # ---------------
        # Initiate logger
        # ---------------
        self.logger = logging.getLogger("MERON")
        self.logger.setLevel(logging.DEBUG)
        hdlr1 = logging.StreamHandler(stream=sys.stdout)
        fmt1 = logging.Formatter(
            "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
        )
        hdlr1.setFormatter(fmt1)
        self.logger.addHandler(hdlr1)

    def _get_class_weights(self, y, neural_net=False):
        '''
        Determine class weights of training data to account if data is skewed
        '''
        class_weights = {}

        w = class_weight.compute_class_weight('balanced', np.unique(y), y)

        for i, val in enumerate(np.unique(y)):
            if neural_net:
                class_weights[i] = w[i]
            else:
                class_weights[val] = w[i]

        return class_weights

    @staticmethod
    def build_model(neurons=128.,
                    dropout=0.,
                    activation='relu',
                    optimizer='adam',
                    input_dim=4098,
                    task_type='classification',
                    n_hidden_layers=1,
                    n_nodes_output=3,
                    reg_val=0.001):
        """
        This function prepares the neural networks for training, using the
        optimized hyperparameters.
        """
        # Add first hidden layer
        conv_model = Sequential()
        if reg_val is None:
            conv_model.add(Dense(neurons, input_dim=input_dim))
        else:
            conv_model.add(Dense(neurons, input_dim=input_dim, kernel_regularizer=l2(reg_val)))
        conv_model.add(Activation(activation))
        conv_model.add(Dropout(dropout))

        # Add additional hidden layers
        for n in range(0, n_hidden_layers-1):
            if reg_val is None:
                conv_model.add(Dense(neurons))
            else:
                conv_model.add(Dense(neurons, kernel_regularizer=l2(reg_val)))
            conv_model.add(Dropout(dropout))
            conv_model.add(Activation(activation))

        # Add output layer
        if task_type == 'classification':
            conv_model.add(Dense(n_nodes_output, activation='softmax'))
            conv_model.compile(loss='categorical_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])

        if task_type == 'regression':
            conv_model.add(Dense(1, activation='linear'))
            conv_model.compile(loss='mean_squared_error',
                               optimizer=optimizer,
                               metrics=[pearsonr])

        return conv_model

    def vgg(self,
            train_x,
            train_y,
            test_x,
            test_y,
            drop_rate,
            out_model_file,
            early_stop_monitor='val_loss',
            early_stop_patience=5,
            n_iter_search=75,
            n_epochs=5000,
            batchsize=512,
            optimizer='Adam',
            task_type='classification'):

        '''Try using the VGG model as a portion of model
        Modify VGG model for producing facial features
           Oxford VGGFace Implementation using Keras Functional Framework v2+
                Very Deep Convolutional Networks for Large-Scale Image Recognition
                K. Simonyan, A. Zisserman
                arXiv:1409.1556

                https://github.com/rcmalli/keras-vggface

        '''
        # Take second fully-connected layer of VGG model
        vgg_model = VGGFace()
        fixed_layers = vgg_model.get_layer('fc6/relu').output

        mod_model = self.extend_existing_model(
            vgg_model.input, fixed_layers, train_x, train_y, test_x, test_y, drop_rate,
            early_stop_monitor=early_stop_monitor, early_stop_patience=early_stop_patience,
            n_iter_search=n_iter_search, n_epochs=n_epochs, batchsize=batchsize,
            optimizer=optimizer, task_type=task_type
        )

        mod_model.save(out_model_file)

        return mod_model

    def encoder(self,
                train_x,
                train_y,
                test_x,
                test_y,
                drop_rate,
                encoder_file,
                out_model_file,
                early_stop_monitor='val_loss',
                early_stop_patience=5,
                n_iter_search=75,
                n_epochs=5000,
                batchsize=512,
                optimizer='Adam',
                task_type='classification'):

        # Take second fully-connected layer of VGG model
        encoder = load_model(encoder_file)
        fixed_layers = encoder.get_layer('Encoder3').output

        mod_model = self.extend_existing_model(
            encoder.input, fixed_layers, train_x, train_y, test_x, test_y, drop_rate,
            early_stop_monitor=early_stop_monitor, early_stop_patience=early_stop_patience,
            n_iter_search=n_iter_search, n_epochs=n_epochs, batchsize=batchsize,
            optimizer=optimizer, task_type=task_type
        )

        mod_model.save(out_model_file)

        return mod_model

    def extend_existing_model(self,
                              model_input,
                              fixed_layers,
                              train_x,
                              train_y,
                              test_x,
                              test_y,
                              drop_rate,
                              early_stop_monitor='val_loss',
                              early_stop_patience=5,
                              n_iter_search=75,
                              n_epochs=5000,
                              batchsize=512,
                              optimizer='Adam',
                              task_type='classification'):

        # Add aditional layers for our training
        x = Dense(256, activation='relu', name='first_mod_layer')(fixed_layers)
        x = Dropout(drop_rate)(x)
        x = Dense(128, activation='relu', name='first_mod_layer')(x)
        x = Dropout(drop_rate)(x)
        x = Dense(32, activation='relu', name='second_mod_layer')(x)
        x = Dropout(drop_rate)(x)

        if task_type == 'classification':
            out = Dense(4, activation='softmax', name='output')(x)
        elif task_type == 'regression':
            out = Dense(1, activation='linear', name='output')(x)

        model_new = Model(model_input, out)

        # Freeze all layers of origina VGG model
        for i, layer in enumerate(model_new.layers):
            if i <= 21:
                layer.trainable = False

        if task_type == 'classification':
            model_new.compile(loss='categorical_crossentropy',
                              optimizer=optimizer,
                              metrics=['accuracy'])
        elif task_type == 'regression':
            model_new.compile(loss='mean_squared_error',
                              optimizer=optimizer,
                              metrics=[pearsonr])

        early_stop = EarlyStopping(monitor=early_stop_monitor, patience=early_stop_patience)

        if self.pred_type == 'classification':
            # train models
            model_new.fit(
                train_x,
                np.array(pd.get_dummies(train_y)),
                epochs=n_epochs,
                batch_size=batchsize,
                validation_data=(test_x, np.array(pd.get_dummies(test_y))),
                callbacks=[early_stop],
                class_weight=self._get_class_weights(train_y, neural_net=True),
                shuffle=True, verbose=1
            )
            self.logger.info('Finished training on convolutional features')

        if self.pred_type == 'regression':
            # train models
            model_new.fit(
                train_x, train_y,
                epochs=n_epochs,
                batch_size=batchsize,
                validation_data=(test_x, np.array(pd.get_dummies(test_y))),
                callbacks=[early_stop],
                shuffle=True, verbose=1
            )
            self.logger.info('Finished training on convolutional features')

        return model_new

    def optimize_hyperparameters(self,
                                 train_x,
                                 train_y,
                                 test_x,
                                 test_y,
                                 param_dist,
                                 early_stop_monitor='val_loss',
                                 early_stop_patience=5,
                                 n_iter_search=75,
                                 n_epochs=5000,
                                 batchsize=512,
                                 val_split=0.2,
                                 num_hid_layers=1,
                                 out_fname=None):
        """
        This function performs a randomized grid seach to optimize the
        hyperparameters of the feedforward neural network and saves a
        dictionary of those parameters. Note that this action only needs
        to be done once; after the optimal parameters are found, we can
        just load in the dictionary.
        """

        # Quit training if validation loss increases for N epochs
        early_stop = EarlyStopping(monitor=early_stop_monitor, patience=early_stop_patience)

        # ------------------------------------------
        # Build models for optimize hyper-parameters
        # Start with initial parameters
        # ------------------------------------------
        init_param_dict = {}
        init_param_dict['neurons'] = 128
        init_param_dict['dropout'] = 0
        init_param_dict['activation'] = 'relu'
        init_param_dict['optimizer'] = 'adam'
        init_param_dict['input_dim'] = 4098
        init_param_dict['task_type'] = 'classification'

        # --------------
        # Classification
        # --------------
        if self.pred_type == 'classification':

            score = make_scorer(geometric_mean_score)
            # Optimizer network with convolutional features
            conv_model = KerasClassifier(
                build_fn=self.build_model,
                epochs=n_epochs,
                batch_size=batchsize,
                validation_data=(test_x, test_y),
                class_weight=self._get_class_weights(train_y, neural_net=True),
                shuffle=True
            )

        # ----------
        # Regression
        # ----------
        if self.pred_type == 'regression':

            score = 'r2'
            conv_model = KerasRegressor(
                build_fn=self.build_model,
                epochs=n_epochs,
                batch_size=batchsize,
                validation_data=(test_x, test_y),
                shuffle=True
            )

        # -------------------
        # Conduct grid search
        # -------------------
        random_search_conv = RandomizedSearchCV(
           conv_model, param_distributions=param_dist, n_iter=n_iter_search, scoring=score,
           fit_params={'callbacks': [early_stop]}
        )

        random_search_conv = random_search_conv.fit(train_x, train_y)

        # ------------------------------
        # save optimized hyperparameters
        # ------------------------------
        if not (out_fname is None):
            np.save(out_fname, random_search_conv.best_params_)

        return random_search_conv.best_params_

    def train_model(self,
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    conv_params,
                    early_stop_monitor='val_loss',
                    early_stop_patience=5,
                    n_iter_search=75,
                    n_epochs=5000,
                    batchsize=512,
                    # val_split=0.2,
                    num_hid_layers=1,
                    out_fname=None):
        """
        This function tunes hyperparameters (or just loads the optimized ones if
        already done), then builds and trains the models using those
        hyperparameters. Returns trained models.
        """

        # only need to tune hyperparemeters once
        # conv_params = np.load(self.config[self.dep]["conv_params_file"]).item()

        conv = self.build_model(**conv_params, n_hidden_layers=num_hid_layers)

        # stop training if valdidation error increases for
        early_stop = EarlyStopping(monitor=early_stop_monitor, patience=early_stop_patience)

        if self.pred_type == 'classification':
            # train models
            conv.fit(
                train_x,
                np.array(pd.get_dummies(train_y)),
                epochs=n_epochs,
                batch_size=batchsize,
                validation_data=(test_x, np.array(pd.get_dummies(test_y))),
                # validation_split=val_split,
                callbacks=[early_stop],
                class_weight=self._get_class_weights(train_y, neural_net=True),
                shuffle=True, verbose=1
            )
            self.logger.info('Finished training on convolutional features')

        if self.pred_type == 'regression':
            # train models
            conv.fit(
                train_x, train_y,
                epochs=n_epochs,
                batch_size=batchsize,
                validation_data=(test_x, np.array(pd.get_dummies(test_y))),
                # validation_split=val_split,
                callbacks=[early_stop],
                shuffle=True, verbose=1
            )
            self.logger.info('Finished training on convolutional features')

        if not (out_fname is None):
            conv.save(out_fname)

        return conv


class MeronMorph(Meron):
    """
    A MORPH database specific instance of MERON. MORPH specific cleaning
    routines.
    """

    def __init__(self, pred_type='classification'):
        Meron.__init__(self, pred_type=pred_type)

    def _load_cnn_features(self, features_dir):

        features = pd.DataFrame()
        for f in os.listdir(features_dir):
            df = pd.read_csv(os.path.join(features_dir, f))
            features = pd.concat([features, df], axis=0)
            print(f + ' loaded')

        features = features.dropna()
        features.iloc[:, 1:] = maxabs_scale(features.iloc[:, 1:])

        return features

    def prep_data(self,
                  features_dir,
                  meta_file,
                  n_params=4096,
                  out_fname=None):

        features = self._load_cnn_features(features_dir)

        # VGG has a feature dimension of 4096 for SoftMax
        var_list_conv = list(np.arange(0, n_params))
        var_list_conv = list(map(str, var_list_conv))

        var_list_conv.extend(['age', 'race_B', 'race_H', 'race_I', 'race_O', 'race_W', 'gender_M'])

        # Read in image meta data
        meta = pd.read_csv(meta_file)
        meta = meta[meta['facial_hair'] != 1]
        meta = meta[meta['glasses'] != 1]
        meta = meta[meta['age'] <= 30]

        # Merge meta data, convolutional features, geometric features
        df = meta.merge(features, on='photo')
        df = df.drop_duplicates(subset='photo')

        # Pull appropriate response variable
        if self.pred_type == 'classification':
            y = df.iloc[:, df.columns == 'bmi_cat']
        else:
            y = df.iloc[:, df.columns == 'bmi']

        # split data in 60-20-20 train-val-test sets
        train_x, test_x, train_y, test_y = train_test_split(
            df, y, test_size=0.2, random_state=42, stratify=y
        )

        train_x = train_x[var_list_conv].values
        test_x = test_x[var_list_conv].values

        # Scale input data
        conv_scaler = StandardScaler().fit(np.concatenate((train_x, test_x), axis=0))

        train_x = conv_scaler.transform(train_x)
        test_x = conv_scaler.transform(test_x)

        train_y = train_y.values.flatten()
        test_y = test_y.values.flatten()

        train_test_set = {}
        train_test_set['train_x'] = train_x
        train_test_set['train_y'] = train_y
        train_test_set['test_y'] = test_y
        train_test_set['test_x'] = test_x

        if not (out_fname is None):
            joblib.dump(conv_scaler, out_fname)

        return train_test_set


class MeronSmart(Meron):
    """
    """

    def __init__(self, pred_type='classification'):
        Meron.__init__(self, pred_type=pred_type)

    def _load_cnn_features(self, features_dir):

        features = pd.DataFrame()
        for f in os.listdir(features_dir):
            df = pd.read_csv(os.path.join(features_dir, f))
            features = pd.concat([features, df], axis=0)
            print(f + ' loaded')

        features = features.dropna()
        features.iloc[:, 1:] = maxabs_scale(features.iloc[:, 1:])

        return features

    def prep_features(self, features_dir, n_params=2048):

        features = self._load_cnn_features(features_dir)

        # vgg has a feature dimension of 4096 for softmax
        var_list_conv = list(map(str, np.arange(0, n_params)))

        features_only = features[var_list_conv]

        train_features, test_features = train_test_split(features_only, test_size=0.2,
                                                         random_state=42)

        return test_features, train_features

    def prep_data(self,
                  features_dir,
                  meta_file,
                  n_params=4096,  # 2048 for resnet50
                  out_fname=None,
                  scaler_flg=False,
                  cname_ind_class='maln_class',
                  cname_ind='wfh',
                  cname_merge='photo_id'):

        features = self._load_cnn_features(features_dir)

        # vgg has a feature dimension of 4096 for softmax
        var_list_conv = list(np.arange(0, n_params))
        var_list_conv = list(map(str, var_list_conv))

        var_list_conv.extend(['age_months', 'gender_male'])

        # Read in image meta data
        meta = pd.read_csv(meta_file)

        # Merge meta data, convolutional features, geometric features
        df = meta.merge(features, on=cname_merge)
        df = df.drop_duplicates(subset=cname_merge)

        # Pull appropriate response variable
        if self.pred_type == 'classification':
            y = df.iloc[:, df.columns == cname_ind_class]
        else:
            y = df.iloc[:, df.columns == cname_ind]

        # split data in 60-20-20 train-val-test sets
        train_x, test_x, train_y, test_y = train_test_split(
            df, y, test_size=0.2, random_state=42, stratify=y
        )

        train_x = train_x[var_list_conv].values
        test_x = test_x[var_list_conv].values

        # Scale input data
        if scaler_flg:
            conv_scaler = StandardScaler().fit(np.concatenate((train_x, test_x), axis=0))

            train_x = conv_scaler.transform(train_x)
            test_x = conv_scaler.transform(test_x)

        train_y = train_y.values.flatten()
        test_y = test_y.values.flatten()

        train_test_set = {}
        train_test_set['train_x'] = train_x
        train_test_set['train_y'] = train_y
        train_test_set['test_y'] = test_y
        train_test_set['test_x'] = test_x

        if not (out_fname is None):
            joblib.dump(conv_scaler, out_fname)

        return train_test_set
