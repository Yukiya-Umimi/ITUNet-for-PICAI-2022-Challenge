__all__ = ['efficientnet-b5']

TASK = 'picai'
NET_NAME = 'efficientnet-b5'
VERSION = 'v0'
DEVICE = '3'
# Must be True when pre-training and inference
PRE_TRAINED = False
# 1,2,3,4
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5
TTA_TIMES = 5

NUM_CLASSES = 3
from utils import get_weight_path,get_weight_list

CSV_PATH = './picai_illness_3c.csv'
CKPT_PATH = './ckpt/{}/{}/fold{}'.format(TASK,VERSION,str(CURRENT_FOLD))
WEIGHT_PATH = get_weight_path(CKPT_PATH)
# WEIGHT_PATH_LIST = None

print(WEIGHT_PATH)

if PRE_TRAINED:
    WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/{}/'.format(TASK,VERSION),choice=[1,2,3,4,5])
else:
    WEIGHT_PATH_LIST = None

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name': NET_NAME,
    'lr': 1e-3,
    'n_epoch': 80,
    'channels': 3,
    'num_classes': NUM_CLASSES,
    'input_shape': (384,384),
    'batch_size': 24,
    'num_workers': 4,
    'device': DEVICE,
    'pre_trained': PRE_TRAINED,
    'weight_path': WEIGHT_PATH,
    'weight_decay': 0,
    'mean': None,
    'std': None,
    'gamma': 0.1,
    'milestones': [30,60],
    'use_fp16':True,
    'external_pretrained':True,
}


# Arguments when perform the trainer
SETUP_TRAINER = {
    'output_dir': './ckpt/{}/{}'.format(TASK,VERSION),
    'log_dir': './log/{}/{}'.format(TASK,VERSION),
    'optimizer': 'AdamW',
    'loss_fun': 'Cross_Entropy',
    'class_weight': None,
    'lr_scheduler': 'MultiStepLR'
}
