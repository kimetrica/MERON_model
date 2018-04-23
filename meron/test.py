from preprocessing import ImagePreProcess, ExtractCNNfeatures, SmartZscores
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

# Create instance of MORPH specific pre-processing
# meron = ImagePreProcess(landmark_file=landmark_file)

# Process images (align, rotate and scale). This processes all the raw images in the raw_img_dir
# It only grabs one face for each image (n_faces=1)
# meron.batch_image_detect_align(raw_img_dir, processed_img_dir, n_faces=1)

# p_img = Image.fromarray(processed_img[0], 'RGB')

# con_feats = ExtractCNNfeatures()
# Extract convolutional features from pre-trained VGG network
# con_feats.extract_batch(processed_img_dir, meta_file, cnn_feature_dir, n=1000)

# Find growth indicators for SMART data
sz = SmartZscores(who_tables_dir)
sz.calc_measures(meta_file, measures=['wfh', 'hfa', 'wfa'])

sz.df_meta.to_csv('/Data/kimetrica/meron/kenya_data/meron_link_data/meron_meta_processed.csv',
    index=False)
