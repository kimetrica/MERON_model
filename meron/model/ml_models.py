import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from imblearn.metrics import geometric_mean_score
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout, Activation
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.models import load_model, Sequential
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
    '''
    '''

    def __init__(self, pred_type='classification'):

        self.pred_type = pred_type
        self.conv_model = None

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

    def build_model(self, conv_param_dict, n_hidden_layers=1):
        """
        This function prepares the neural networks for training, using the
        optimized hyperparameters.
        """
        # Add first hidden layer
        conv_model = Sequential()
        conv_model.add(Dense(conv_param_dict['neurons'], input_dim=conv_param_dict['input_dim']))
        conv_model.add(Activation(conv_param_dict['activation']))
        conv_model.add(Dropout(conv_param_dict['dropout']))

        # Add additional hidden layers
        for n in range(0, n_hidden_layers-1):
            conv_model.add(Dense(conv_param_dict['neurons']))
            conv_model.add(Dropout(conv_param_dict['dropout']))
            conv_model.add(Activation(conv_param_dict['activation']))

        # Add output layer
        if self.pred_type == 'classification':
            conv_model.add(Dense(4, activation='softmax'))
            conv_model.compile(loss='categorical_crossentropy',
                               optimizer=conv_param_dict['optimizer'],
                               metrics=['accuracy'])

        if self.pred_type == 'regression':
            conv_model.add(Dense(1, activation='linear'))
            conv_model.compile(loss='mean_squared_error',
                               optimizer=conv_param_dict['optimizer'],
                               metrics=[pearsonr])

        self.conv_model = conv_model

        return conv_model

    def optimize_hyperparameters(self,
                                 train_x_conv,
                                 train_y,
                                 param_dist,
                                 early_stop_monitor='val_los',
                                 early_stop_patience=5,
                                 n_iter_search=75,
                                 n_epochs=5000,
                                 batchsize=128,
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
        init_param_dict['input_dim'] = 4103
        init_param_dict['task_type'] = 'classification'

        conv_model_tune = self.build_model(init_param_dict, n_hidden_layers=num_hid_layers)

        # --------------
        # Classification
        # --------------
        if self.pred_type == 'classification':

            score = make_scorer(geometric_mean_score)
            # Optimizer network with convolutional features
            conv_model = KerasClassifier(
                build_fn=conv_model_tune,
                epochs=n_epochs,
                batch_size=batchsize,
                validation_split=val_split,
                class_weight=self._get_class_weights(train_y, neural_net=True),
                shuffle=True
            )

        # ----------
        # Regression
        # ----------
        if self.pred_type == 'regression':

            score = 'r2'
            conv_model = KerasRegressor(
                build_fn=conv_model_tune,
                epochs=n_epochs,
                batch_size=batchsize,
                validation_split=val_split,
                shuffle=True
            )

        # -------------------
        # Conduct grid search
        # -------------------
        random_search_conv = RandomizedSearchCV(
           conv_model, param_distributions=param_dist, n_iter=n_iter_search, scoring=score,
           fit_params={'callbacks': [early_stop]}
        )

        random_search_conv = random_search_conv.fit(train_x_conv, train_y)

        # ------------------------------
        # save optimized hyperparameters
        # ------------------------------
        if not (out_fname is None):
            np.save(out_fname, random_search_conv.best_params_)

        return random_search_conv.best_params_

    def train_model(self,
                    train_x,
                    train_y,
                    conv_params,
                    early_stop_monitor='val_los',
                    early_stop_patience=5,
                    n_iter_search=75,
                    n_epochs=5000,
                    batchsize=128,
                    val_split=0.2,
                    num_hid_layers=1,
                    out_fname=None):
        """
        This function tunes hyperparameters (or just loads the optimized ones if
        already done), then builds and trains the models using those
        hyperparameters. Returns trained models.
        """

        # only need to tune hyperparemeters once
        # conv_params = np.load(self.config[self.dep]["conv_params_file"]).item()

        conv = self.build_models(conv_params, n_hidden_layers=num_hid_layers)

        # stop training if valdidation error increases for
        early_stop = EarlyStopping(monitor=early_stop_monitor, patience=early_stop_patience)

        if self.pred_type == 'classification':
            # train models
            conv.fit(
                train_x,
                np.array(pd.get_dummies(train_y)),
                epochs=n_epochs,
                batch_size=batchsize,
                validation_split=val_split,
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
                validation_split=val_split,
                callbacks=[early_stop],
                shuffle=True, verbose=1
            )
            self.logger.info('Finished training on convolutional features')

        self.conv_model = conv

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
                  geom_file,
                  meta_file,
                  out_fname=None):

        features = self._load_cnn_features(features_dir)

        # VGG has a feature dimension of 4096 for SoftMax
        var_list_conv = list(np.arange(0, 4096))
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
            df, y, test_size=0.2, random_state=42
        )

        train_x_conv = train_x[var_list_conv].values
        test_x_conv = test_x[var_list_conv].values

        # Scale input data
        conv_scaler = StandardScaler().fit(np.concatenate((train_x_conv, test_x_conv), axis=0))

        train_x_conv = conv_scaler.transform(train_x_conv)
        test_x_conv = conv_scaler.transform(test_x_conv)

        train_y = train_y.values.flatten()
        test_y = test_y.values.flatten()

        train_test_set = {}
        train_test_set['train_x_conv'] = train_x_conv
        train_test_set['train_y'] = train_y
        train_test_set['test_y'] = test_y
        train_test_set['test_x_conv'] = test_x_conv

        if not (out_fname is None):
            joblib.dump(conv_scaler, out_fname)

        return train_test_set
