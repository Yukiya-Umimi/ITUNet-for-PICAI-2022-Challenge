import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np

from classification.config import (FOLD_NUM, INIT_TRAINER, SETUP_TRAINER, TASK,
                                   VERSION)
from classification.data_utils.csv_reader import csv_reader_single
from classification.run import get_cross_validation, get_parameter_number
from classification.trainer import Classifier
from classification.utils import get_weight_path


def main():
    """Training method from Swangeese team for PI-CAI Challenge."""
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--workdir', type=str, default="/workdir")
    parser.add_argument('--preprocesseddir', type=str, default=os.environ.get('SM_CHANNEL_PREPROCESSED', "/input/preprocessed"))
    parser.add_argument('--outputdir', type=str, default=os.environ.get('SM_MODEL_DIR', "/output"))
    parser.add_argument('--checkpointsdir', type=str, default="/checkpoints")
    parser.add_argument('--folds', type=int, nargs="+", default=(1, 2, 3, 4, 5),
                        help="Folds to train. Default: 1 2 3 4 5")

    args, _ = parser.parse_known_args()

    # paths
    workdir = Path(args.workdir)
    output_dir = Path(args.outputdir)
    checkpoints_dir = Path(args.checkpointsdir)
    preprocessed_dir = Path(args.preprocesseddir)

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # descibe input data
    print(f"workdir: {workdir}")
    print(f"output_dir: {output_dir}")

    # Train classification model
    label_dict = {}
    INIT_TRAINER['device'] = "0"

    # Set data path & classifier
    pre_csv_path = preprocessed_dir / "classification" / "picai_illness_3c.csv"
    pre_label_dict = csv_reader_single(pre_csv_path, key_col='id', value_col='label')

    # update paths to preprocessed data directory
    pre_label_dict = {
        preprocessed_dir / Path(path).relative_to("/output"): label
        for path, label in pre_label_dict.items()
    }

    label_dict.update(pre_label_dict)

    # Training with cross validation
    ###############################################
    path_list = list(label_dict.keys())

    loss_list = []
    acc_list = []

    for current_fold in args.folds:
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
        SETUP_TRAINER['output_dir'] = checkpoints_dir / f'ckpt/{TASK}/{VERSION}'
        SETUP_TRAINER['log_dir'] = checkpoints_dir / f'log/{TASK}/{VERSION}'
        #SETUP_TRAINER['seed'] = count

        start_time = time.time()
        val_loss, val_acc = classifier.trainer(**SETUP_TRAINER)
        loss_list.append(val_loss)
        acc_list.append(val_acc)

        print('run time:%.4f' % (time.time()-start_time))

    print("Average loss is %f, average acc is %f" %
            (np.mean(loss_list), np.mean(acc_list)))

    # Export trained models
    for fold in args.folds:
        ckpt_dir = checkpoints_dir / f'ckpt/{TASK}/{VERSION}' / f'fold{fold}'
        ckpt_path = get_weight_path(ckpt_dir)
        dst = output_dir / f"ckpt/{TASK}/{VERSION}/fold{fold}.pth"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ckpt_path, dst)


if __name__ == '__main__':
    main()
