import os
import glob
from utils import get_weight_path

TRANSFORMER_DEPTH = 24
VERSION = 'itunet_d24'

PHASE = 'seg'   # 'seg' or 'detect'
NUM_CLASSES = 2 if 'seg' in PHASE else 3

DEVICE = '3'
# True if use internal pre-trained model
# Must be True when pre-training and inference
PRE_TRAINED = False
# True if use resume model
CKPT_POINT = False

FOLD_NUM = 5
# [1-FOLD_NUM]
CURRENT_FOLD = 5
GPU_NUM = len(DEVICE.split(','))

#--------------------------------- mode and data path setting
PATH_DIR = './dataset/segdata/data_2d'
PATH_LIST = glob.glob(os.path.join(PATH_DIR,'*.hdf5'))
PATH_AP = './dataset/segdata/data_3d'
AP_LIST = glob.glob(os.path.join(PATH_AP,'*.hdf5'))
#--------------------------------- 

CKPT_PATH = './new_ckpt/{}/{}/fold{}'.format(PHASE,VERSION,str(CURRENT_FOLD))
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

#you could set it for your device
INIT_TRAINER = {
  'num_classes':NUM_CLASSES, 
  'n_epoch':160,
  'batch_size':24,  
  'num_workers':4,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'ckpt_point':CKPT_POINT,
  'weight_path':WEIGHT_PATH,
  'use_fp16':False,
  'transformer_depth': TRANSFORMER_DEPTH
 }
#---------------------------------

SETUP_TRAINER = {
  'output_dir':'./new_ckpt/{}/{}'.format(PHASE,VERSION),
  'log_dir':'./new_log/{}/{}'.format(PHASE,VERSION),
  'phase':PHASE,
  }
