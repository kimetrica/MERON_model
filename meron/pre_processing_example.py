
if __name__ == "__main__":


    in_dir = '/media/ebaumer/backup/Data/MORPH/commercial/Album2'
    out_dir = '/media/ebaumer/backup/Data/morph_processed'
    con_feat_dir = '/home/Data/kimetrica/meron/features_fc6'
    geom_feat_dir = '/home/Data/kimetrica/meron'

    morph_meta_file = '/media/ebaumer/backup/Data/MORPH/commercial/MORPH_Album2_Commercial.csv'
    # morph_meta_ofile = '/media/ebaumer/backup/Data/morph_processed/'
    morph_meta_ofile = '/Data/kimetrica/meron/MORPH_Album2_Commercial_processed.csv'
    # Create instance
    morph = MorphPreProcess()
    # Process images (align, rotate and scale)
    # morph.batch_image_detect_align(in_dir, out_dir)

    # Pre-process MORPH meta-data file
    # morph.preprocess_morph_meta(morph_meta_file, morph_meta_ofile)

    # Extract convolutional features from vgg
    con_feats = ExtractCNNfeatures()
    con_feats.extract_batch(img_dir=out_dir, processed_data_file=morph_meta_ofile)
