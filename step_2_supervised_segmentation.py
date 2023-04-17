import argparse
import os
import shutil
import time
from pathlib import Path

from segmentation.config import (FOLD_NUM, INIT_TRAINER, PHASE, SETUP_TRAINER,
                                 VERSION)
from segmentation.run import (get_cross_validation_by_sample,
                              get_parameter_number)
from segmentation.trainer import SemanticSeg
from segmentation.utils import get_weight_path


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

    # set training parameters
    INIT_TRAINER['device'] = "0"

    path_list = (preprocessed_dir / "segmentation/segdata/data_2d").glob("*.hdf5")
    path_list = [str(path) for path in path_list]
    ap_list = (preprocessed_dir / "segmentation/segdata/data_2d").glob("*.hdf5")
    ap_list = [str(path) for path in ap_list]

    # Training with cross validation
    for current_fold in args.folds:
        print("=== Training Fold ", current_fold, " ===")
        segnetwork = SemanticSeg(**INIT_TRAINER)
        print(get_parameter_number(segnetwork.net))
        train_path, val_path = get_cross_validation_by_sample(path_list, FOLD_NUM, current_fold)
        train_AP, val_AP = get_cross_validation_by_sample(ap_list, FOLD_NUM, current_fold)
        SETUP_TRAINER['train_path'] = train_path
        SETUP_TRAINER['val_path'] = val_path
        SETUP_TRAINER['val_ap'] = val_AP
        SETUP_TRAINER['cur_fold'] = current_fold
        SETUP_TRAINER['output_dir'] = checkpoints_dir / f'ckpt/{PHASE}/{VERSION}'
        SETUP_TRAINER['log_dir'] = checkpoints_dir / f'log/{PHASE}/{VERSION}'
        start_time = time.time()
        segnetwork.trainer(**SETUP_TRAINER)

        print('run time:%.4f' % (time.time() - start_time))

    # Export trained models
    for fold in args.folds:
        ckpt_dir = checkpoints_dir / f'ckpt/{PHASE}/{VERSION}' / f'fold{fold}'
        ckpt_path = get_weight_path(ckpt_dir)
        dst = output_dir / f"ckpt/{PHASE}/{VERSION}/fold{fold}.pth"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ckpt_path, dst)


if __name__ == '__main__':
    main()
