
# Flags
train_model_flg = True
tune_hyperparams_flg = True
save_model_flg = True

# Files and directories
base_dir = '/Data/kimetrica/meron'
features_dir = '/Data/kimetrica/meron/features_fc6'
model_dir = '/Data/kimetrica/meron/models'
meta_file = 'MORPH_Album2_Commercial_processed.csv'  # file w/ BMI data
conv_model_file = 'conv_model.h5'  # trained convolutional model
conv_params_file = 'conv_params.npy'  # tuned parameters for convolutional model

classes = ['normal', 'obese', 'overweight', 'underweight']

# Instantiate class
meron = MeronMorph()

# Prep train/test data
data_tt = meron.prep_data(features_dir, os.path.join(base_dir, geom_file),
                          os.path.join(base_dir, meta_file))

# Train nn with option to tune hyper parameters
if train_model_flg:
    conv, geom = meron.train_model(
        data_tt["train_x_geom"], data_tt["train_x_conv"], data_tt["train_y"],
        tune_hyperparams=tune_hyperparams_flg, save_model=save_model_flg
        )

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
