import os
import numpy as np
import pandas as pd
from preprocessing import ImagePreProcess, ExtractCNNfeatures, SmartZscores
from model import MeronSmart


# This directory contains the original raw SMART images
raw_img_dir = '/data/meron/kenya_data/meron_photos'
morph_raw_img_dir = '/data/MORPH/commercial/Album2'
faces_wild_raw_img_dir = '/data/kimetrica/meron/face_imgs'

# This is the directory to write pre-processed images
# processed_img_dir = '/Data/kimetrica/meron/kenya_data/meron_photos_processed/'
processed_img_dir = '/data/meron/aligned_face_imgs'
smart_processed_dir='/data/meron/kenya_data/meron_photos_processed'
smart_augmented_dir='/data/meron/kenya_data/photo_aug(2)'
# This is the directory to write the CNN feature files
# feature_dir = '/data/meron/features_resnet'
feature_dir ='/data/meron/features_vgg16'
smart_feature_dir_vgg16='/data/meron/smart_features_vgg16'
smart_feature_augmented ='/data/meron/features_resnet_augmented/sam_mam_upsampled'
# This is the file for Dlib trained facial landmark detection model. Used for aligning facial image
landmark_file = '/home/kimetrica/code/kimetrica/meron_gh/data/shape_predictor_68_face_landmarks.dat'

# This is the meta file for the SMART/MERON data
# meta_file = '/Data/kimetrica/meron/kenya_data/meron_link_data/all_areas.csv'
# meta_file = '/Data/kimetrica/meron/aligned_files.csv'
meta_file = '/data/meron/updated_features.csv'
meron_data_file='/data/meron/kenya_data/meron_images(clahe).csv'

# This is the directory for the who zscore data tables
who_tables_dir = '/home/kimetrica/Code/kimetrica/meron_gh/data'

# Processed meta-data file
processed_meta_file = '/data/meron/kenya_data/meron_link_data/meron_meta_processed.csv'
morph_processed_meta_file = '/data/meron/MORPH_Album2_Commercial_processed.csv'
# This is the directory to write the CNN feature files
model_encoder = '/data/meron/models/encoder_cnn_vgg16.h5'

###folder for autoencoder features_resnet
smart_encoder_feat_dir='/data/meron/smart_features_encoder'

# Number of features of from facial recognition model
nparams = 4096

# -----------------------
# Detect and align images
# -----------------------
# Create instance of MORPH specific pre-processing
# meron = ImagePreProcess(landmark_file=landmark_file)

# Process images (align, rotate and scale). This processes all the raw images in the raw_img_dir
# It only grabs one face for each image (n_faces=1)
# meron.batch_image_detect_align(faces_wild_raw_img_dir, processed_img_dir, n_faces=1)

# --------------------
# Extract CNN features
# --------------------
con_feats = ExtractCNNfeatures()
# Extract convolutional features from pre-trained VGG network
# con_feats.extract_batch(processed_img_dir, meta_file, feature_dir, model_type='vgg16', n=5000)
con_feats.extract_batch(smart_augmented_dir, meron_data_file, smart_feature_augmented, model_type='resnet50',
                         n=1000)

# Read in VGG CNN features
"""
meron_smart = MeronSmart()

encoder_model = con_feats.train_auto_encoder(train_x, test_x, input_dim=nparams,
                                           out_model_file=model_encoder)

#### load the tuned encoder_model, and extract features from MERON data, 3843 samples
#test_features, train_feature, features_only = meron_smart.prep_features(smart_feature_dir)
#pretrained_encoder = meron_smart.encoder_feature_test(model_encoder)
#predicted_features = pretrained_encoder.predict(features_only)

#predicted_df = pd.DataFrame(predicted_features)
#fname = 'smart_encoder_features_1.csv'
#predicted_df.to_csv(os.path.join(smart_encoder_feat_dir, fname), index=False)
### save predicted features to folder

# --------------------------------------
# We are going to train the auto-encoder
# on the larger MORPH dataset
# --------------------------------------
# Prep data from MORPH features
# nparams = 2048
data_tt = meron_smart.prep_data(resnet_feature_dir, processed_meta_file, n_params=nparams)

# Remove the last two columns of training and testing explanatory variables
data_tt['train_x'] = np.delete(data_tt['train_x'], np.s_[-2::], 1)
data_tt['test_x'] = np.delete(data_tt['test_x'], np.s_[-2::], 1)

encoder_model = con_feats.train_auto_encoder(data_tt['train_x'], data_tt['test_x'],
                                             input_dim=nparams, out_model=model_encoder)

data_tt = meron_smart.prep_data(resnet_feature_dir, processed_meta_file, n_params=nparams)

# Remove the last two columns of training and testing explanatory variables
# data_tt['train_x'] = np.delete(data_tt['train_x'], np.s_[-2::], 1)
# data_tt['test_x'] = np.delete(data_tt['test_x'], np.s_[-2::], 1)

# encoder_model = con_feats.deep_auto_encoder(data_tt['train_x'], data_tt['test_x'], input_dim=2048,
#                                             out_model=model_encoder)

# Find growth indicators for SMART data
sz = SmartZscores(who_tables_dir, meta_file)
# sz.calc_measures(measures=['wfh', 'hfa', 'wfa'])
# sz.classify_malnutrition()
# sz.cat_encoding()

# sz.write_processed_meta(processed_meta_file)
"""
