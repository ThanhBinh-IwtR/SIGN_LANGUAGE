'''
Our project base on a Kaggle Competion, It have limit the size of time using for training.
Preprocessing Layer is also in Training Process, for Tflite model. 
'''
import numpy as np
import tensorflow as tf

DIR_NAME = 'statics'
COLUMNS0 = np.load(DIR_NAME+'/COLUMNS0.npy')
N_COLS0 = N_COLS = len(COLUMNS0)
N_TARGET_FRAMES = 128

