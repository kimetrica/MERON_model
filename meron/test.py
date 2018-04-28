import numpy as np
import os
import pandas as pd
from preprocessing import ImagePreProcess, ExtractCNNfeatures, SmartZscores
from model import MeronSmart
from PIL import Image


# This directory contains the original raw SMART images
raw_img_dir = '/Data/kimetrica/meron/kenya_data/meron_photos'

# This is the directory to write pre-processed images
processed_img_dir = '/Data/kimetrica/meron/kenya_data/meron_photos_processed/'

# This is the directory to write the CNN feature files
cnn_feature_dir = '/Data/kimetrica/meron/features_fc6'

# This is the file for Dlib trained facial landmark detection model. Used for aligning facial image
landmark_file = '/home/ebaumer/Code/kimetrica/meron_gh/data/shape_predictor_68_face_landmarks.dat'

# This is the meta file for the SMART/MERON data
meta_file = '/Data/kimetrica/meron/kenya_data/meron_link_data/all_areas.csv'

# This is the directory for the who zscore data tables
who_tables_dir = '/home/ebaumer/Code/kimetrica/meron_gh/data'

# Processed meta-data file
processed_meta_file = '/Data/kimetrica/meron/kenya_data/meron_link_data/meron_meta_processed.csv'

# This is the directory to write the CNN feature files
model_dir = '/Data/kimetrica/meron/models'

# Create instance of MORPH specific pre-processing
# meron = ImagePreProcess(landmark_file=landmark_file)

# Process images (align, rotate and scale). This processes all the raw images in the raw_img_dir
# It only grabs one face for each image (n_faces=1)
# meron.batch_image_detect_align(raw_img_dir, processed_img_dir, n_faces=1)

# p_img = Image.fromarray(processed_img[0], 'RGB')

con_feats = ExtractCNNfeatures()
# Extract convolutional features from pre-trained VGG network
# con_feats.extract_batch(processed_img_dir, meta_file, cnn_feature_dir, n=1000)

# Read in VGG CNN features
meron_smart = MeronSmart()
data_tt = meron_smart.prep_data(cnn_feature_dir, processed_meta_file)

# Remove the last two columns of training and testing explanatory variables
data_tt['train_x'] = np.delete(data_tt['train_x'], np.s_[-2::], 1)
data_tt['test_x'] = np.delete(data_tt['test_x'], np.s_[-2::], 1)

encoder_model = con_feats.deep_auto_encoder(data_tt['train_x'], data_tt['test_x'], input_dim=4096,
                                            out_dir=model_dir)

# Find growth indicators for SMART data
import ipdb; ipdb.set_trace()  # breakpoint eeeac112 //
sz = SmartZscores(who_tables_dir, meta_file)
# sz.calc_measures(measures=['wfh', 'hfa', 'wfa'])
# sz.classify_malnutrition()
# sz.cat_encoding()

# sz.write_processed_meta(processed_meta_file)
