import argparse
import os
import random
import time

from segmentation.config import (AP_LIST, CURRENT_FOLD, FOLD_NUM, INIT_TRAINER,
                                 PATH_LIST, SETUP_TRAINER, VERSION)
from segmentation.trainer import SemanticSeg


def get_cross_validation_by_sample(path_list, fold_num, current_fold):

    sample_list = list(set([os.path.basename(case).split('_')[0] for case in path_list]))
    sample_list.sort()
    print('number of sample:',len(sample_list))
    _len_ = len(sample_list) // fold_num

    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(sample_list[start_index:])
        train_id.extend(sample_list[:start_index])
    else:
        validation_id.extend(sample_list[start_index:end_index])
        train_id.extend(sample_list[:start_index])
        train_id.extend(sample_list[end_index:])

    train_path = []
    validation_path = []
    for case in path_list:
        if os.path.basename(case).split('_')[0] in train_id:
            train_path.append(case)
        else:
            validation_path.append(case)

    random.shuffle(train_path)
    random.shuffle(validation_path)
    print("Train set length ", len(train_path),
          "Val set length", len(validation_path))
    return train_path, validation_path

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train-cross',
                        choices=["train", 'train-cross'],
                        help='choose the mode',
                        type=str)
    args = parser.parse_args()

    # Set data path & segnetwork
    if args.mode != 'train-cross':
        segnetwork = SemanticSeg(**INIT_TRAINER)
        print(get_parameter_number(segnetwork.net))
    path_list = PATH_LIST
    # Training
    ###############################################
    if args.mode == 'train-cross':
        for current_fold in range(1, FOLD_NUM + 1):
            print("=== Training Fold ", current_fold, " ===")
            segnetwork = SemanticSeg(**INIT_TRAINER)
            print(get_parameter_number(segnetwork.net))
            train_path, val_path = get_cross_validation_by_sample(path_list, FOLD_NUM, current_fold)
            train_AP, val_AP = get_cross_validation_by_sample(AP_LIST, FOLD_NUM, current_fold)
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['val_ap'] = val_AP
            SETUP_TRAINER['cur_fold'] = current_fold
            start_time = time.time()
            segnetwork.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))


    if args.mode == 'train':
        train_path, val_path = get_cross_validation_by_sample(path_list, FOLD_NUM,CURRENT_FOLD)
        train_AP, val_AP = get_cross_validation_by_sample(AP_LIST, FOLD_NUM, CURRENT_FOLD)
        SETUP_TRAINER['train_path'] = train_path
        SETUP_TRAINER['val_path'] = val_path
        SETUP_TRAINER['val_ap'] = val_AP
        SETUP_TRAINER['cur_fold'] = CURRENT_FOLD
		
        start_time = time.time()
        segnetwork.trainer(**SETUP_TRAINER)

        print('run time:%.4f' % (time.time() - start_time))
    ###############################################
