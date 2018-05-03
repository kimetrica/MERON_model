import numpy as np
import os
import pandas as pd
from preprocessing import ImagePreProcess, ExtractCNNfeatures, SmartZscores
from model import MeronSmart
from PIL import Image


# This directory contains the original raw SMART images
raw_img_dir = '/Data/kimetrica/meron/kenya_data/meron_photos'
morph_raw_img_dir = '/Data/Data/MORPH/commercial/Album2'
faces_wild_raw_img_dir = '/Data/kimetrica/meron/face_imgs'

# This is the directory to write pre-processed images
# processed_img_dir = '/Data/kimetrica/meron/kenya_data/meron_photos_processed/'
processed_img_dir = '/Data/kimetrica/meron/aligned_face_imgs'

# This is the directory to write the CNN feature files
morph_feature_dir = '/Data/kimetrica/meron/features_fc6'
resnet_feature_dir = '/Data/kimetrica/meron/features_resnet'

# This is the file for Dlib trained facial landmark detection model. Used for aligning facial image
landmark_file = '/home/ebaumer/Code/kimetrica/meron_gh/data/shape_predictor_68_face_landmarks.dat'

# This is the meta file for the SMART/MERON data
# meta_file = '/Data/kimetrica/meron/kenya_data/meron_link_data/all_areas.csv'
# meta_file = '/Data/kimetrica/meron/aligned_files.csv'
meta_file = '/Data/kimetrica/meron/updated_features.csv'

# This is the directory for the who zscore data tables
who_tables_dir = '/home/ebaumer/Code/kimetrica/meron_gh/data'

# Processed meta-data file
processed_meta_file = '/Data/kimetrica/meron/kenya_data/meron_link_data/meron_meta_processed.csv'
morph_processed_meta_file = '/Data/kimetrica/meron/MORPH_Album2_Commercial_processed.csv'

# This is the directory to write the CNN feature files
model_encoder = '/Data/kimetrica/meron/models/encoder_cnn.h5'

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
# con_feats.extract_batch(processed_img_dir, meta_file, resnet_feature_dir, model_type='resnet50',
#                         n=5000)
# con_feats.extract_batch(processed_img_dir, meta_file, resnet_feature_dir, model_type='resnet50',
#                         n=1000)

# Read in VGG CNN features
meron_smart = MeronSmart()
# --------------------------------------
# We are going to train the auto-encoder
# on the larger MORPH dataset
# --------------------------------------
# Prep data from MORPH features
data_tt = meron_smart.prep_data(morph_feature_dir, processed_meta_file, n_params=2048)

# Remove the last two columns of training and testing explanatory variables
data_tt['train_x'] = np.delete(data_tt['train_x'], np.s_[-2::], 1)
data_tt['test_x'] = np.delete(data_tt['test_x'], np.s_[-2::], 1)

encoder_model = con_feats.train_auto_encoder(data_tt['train_x'], data_tt['test_x'], input_dim=2048,
                                             out_model=model_encoder)

data_tt = meron_smart.prep_data(resnet_feature_dir, processed_meta_file, n_params=2048)


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
