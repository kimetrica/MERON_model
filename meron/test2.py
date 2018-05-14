import os
import numpy as np
import pandas as pd
from preprocessing import ImagePreProcess, ExtractCNNfeatures, SmartZscores
from model import MeronSmart




raw_img_dir = '/data/meron/kenya_data/temp' 
processed_img_dir ='/data/meron/kenya_data/meron_photos_processed'

landmark_file = '/home/kimetrica/code/MERON/data/shape_predictor_68_face_landmarks.dat'
meron = ImagePreProcess(landmark_file=landmark_file)

meron.batch_image_detect_align(raw_img_dir, processed_img_dir, n_faces=1)  