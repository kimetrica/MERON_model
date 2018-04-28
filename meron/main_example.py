from keras.models import load_model
import numpy as np
import os
from model import MeronSmart
from model import plot_confusion_matrix

# Flags
train_model_flg = True
tune_hyperparams_flg = False
save_model_flg = True

# Files and directories
base_dir = '/Data/kimetrica/meron/kenya_data/meron_link_data'
features_dir = '/Data/kimetrica/meron/features_fc6'
model_dir = '/Data/kimetrica/meron/models'
meta_file = 'meron_meta_processed.csv'
hyper_param_file = '/Data/kimetrica/meron/hyperparams/'

classes = ['normal', 'moderate acute malnutrition', 'severe acute malnutrition']

# hyperparameters to optimizer over
param_dist = {'neurons': [8, 16, 32, 64, 128, 256],
              'dropout': [0, 0.25, 0.5, 0.75, 0.9],
              'activation': ["relu", "elu", "tanh"],
              'optimizer': ["adam", "nadam", "adadelta", "rmsprop"],
              'task_type': ["classification"]}

tuned_params = {'neurons': 64,
                'dropout': 0.25,
                'activation': "relu",
                'optimizer': "adam",
                'task_type': "classification"}

# Instantiate class
meron = MeronSmart()

# Prep train/test data
data_tt = meron.prep_data(features_dir,
                          os.path.join(base_dir, meta_file),
                          out_fname=None,
                          cname_ind_class='maln_class',
                          cname_ind='wfh',
                          cname_merge='photo_id')

# Train nn with option to tune hyper parameters
if tune_hyperparams_flg:
    tuned_params = meron.optimize_hyperparameters(
        data_tt['train_x'], data_tt['train_y'],
        data_tt['test_x'], data_tt['test_y'],
        param_dist,
    )
# else:
#     tuned_params = np.load(hyper_param_file).item()

if train_model_flg:
    conv = meron.train_model(data_tt["train_x"],
                             data_tt["train_y"],
                             data_tt["test_x"],
                             data_tt["test_y"],
                             tuned_params,
                             out_fname=save_model_flg)

else:
    conv = load_model(os.path.join(model_dir, "conv_model_classification.h5"))

# Produce predicted probabilities for training and test examples
conv_probs_test = conv.predict(data_tt["test_x_conv"])

# Convert probabilities to actual predictions
conv_test = np.argmax(conv_probs_test, axis=1)

# le = sk.preprocessing.LabelEncoder()
# le.fit(classes)
# y_pred = np.argmax(conv_probs_test, axis=1)
# y_pred = le.inverse_transform(y_pred)

# Evaluate results
plot_confusion_matrix(data_tt["test_y"], conv_test, classes, normalize=True,
                      savefig='/Data/kimetrica/meron/figs/cm.jpg')
