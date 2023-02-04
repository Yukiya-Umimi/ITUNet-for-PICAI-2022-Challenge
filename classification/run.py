import argparse
from trainer import Classifier
from data_utils.csv_reader import csv_reader_single
from config import INIT_TRAINER, SETUP_TRAINER, FOLD_NUM, CSV_PATH

import time
import numpy as np

KEY = {
    'picai':['image_name','class_id'],
}

SHIFT = {
    'picai':0
}

def get_cross_validation(path_list, fold_num, current_fold):

    _len_ = len(path_list) // fold_num

    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(path_list[start_index:])
        train_id.extend(path_list[:start_index])
    else:
        validation_id.extend(path_list[start_index:end_index])
        train_id.extend(path_list[:start_index])
        train_id.extend(path_list[end_index:])

    print("Train set length:", len(train_id),
          "Val set length:", len(validation_id))
    return train_id, validation_id


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='train-cross', choices=["train-cross"],
                        help='choose the mode', type=str)
    args = parser.parse_args()
    
    label_dict = {}
    # Set data path & classifier
    
    pre_csv_path = CSV_PATH
    pre_label_dict = csv_reader_single(pre_csv_path, key_col='id', value_col='label')

    label_dict.update(pre_label_dict)

    # Training with cross validation
    ###############################################
    if args.mode == 'train-cross':
        path_list = list(label_dict.keys())

        loss_list = []
        acc_list = []

        for current_fold in range(1, FOLD_NUM+1):
            print("=== Training Fold ", current_fold, " ===")
            classifier = Classifier(**INIT_TRAINER)

            if current_fold == 0:
                print(get_parameter_number(classifier.net))

            train_path, val_path = get_cross_validation(
                path_list, FOLD_NUM, current_fold)

            print("dataset length is %d"%len(train_path))
            print("dataset length is %d"%len(val_path))

            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['label_dict'] = label_dict
            SETUP_TRAINER['cur_fold'] = current_fold
            #SETUP_TRAINER['seed'] = count

            start_time = time.time()
            val_loss, val_acc = classifier.trainer(**SETUP_TRAINER)
            loss_list.append(val_loss)
            acc_list.append(val_acc)

            print('run time:%.4f' % (time.time()-start_time))

        print("Average loss is %f, average acc is %f" %
              (np.mean(loss_list), np.mean(acc_list)))
