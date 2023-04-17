import argparse
import os
from pathlib import Path

from classification.cls_data import predict_test5c
from segmentation.predict_2d import postprecess, save_npy, vote_dir


def main(taskname="Task2203_picai_baseline"):
    """Training method from Swangeese team for PI-CAI Challenge."""
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--workdir', type=str, default="/workdir")
    parser.add_argument('--imagesdir', type=str, default=os.environ.get('SM_CHANNEL_IMAGES', "/input/images"))
    parser.add_argument('--labelsdir', type=str, default=os.environ.get('SM_CHANNEL_LABELS', "/input/picai_labels"))
    parser.add_argument('--preprocesseddir', type=str, default=os.environ.get('SM_CHANNEL_PREPROCESSED', "/input/preprocessed"))
    parser.add_argument('--outputdir', type=str, default=os.environ.get('SM_MODEL_DIR', "/output"))
    parser.add_argument('--checkpointsdir', type=str, default="/checkpoints")
    parser.add_argument('--folds', type=int, nargs="+", default=(0, 1, 2, 3, 4),
                        help="Folds to train. Default: 0 1 2 3 4")

    args, _ = parser.parse_known_args()

    # paths
    workdir = Path(args.workdir)
    images_dir = Path(args.imagesdir)
    labels_dir = Path(args.labelsdir)
    output_dir = Path(args.outputdir)
    checkpoints_dir = Path(args.checkpointsdir)
    preprocessed_dir = Path(args.preprocesseddir)

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # descibe input data
    print(f"workdir: {workdir}")
    print(f"images_dir: {images_dir}")
    print(f"labels_dir: {labels_dir}")
    print(f"output_dir: {output_dir}")

    # Perform inference with the classification model
    predict_test5c(
        weight_path: str = '/opt/cls_algorithm/weights/',
        base_dir: str = 'path/to/nnUNet_test_data',
        csv_save_path: str = 'test_3c.csv',
    )

    # Perform inference with the segmentation model
    data_path = preprocessed_dir / "nnUNet_test_data"
    outdir = './segout/segmentation_result'
    config = Config()
    for fold in range(1,6):
        print('****fold%d****'%fold)
        config.fold = fold
        config.ckpt_path = f'./new_ckpt/seg/{config.version}/fold{str(fold)}'
        save_dir = f'./segout/{config.version}/fold{str(fold)}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(data_path)]
        pathlist = list(set(pathlist))

        for path in pathlist:
            pred = predict_process(path,config,data_path)
            print(pred.shape)
            np.save(os.path.join(save_dir,path+'.npy'),pred)
    vote_dir()
    postprecess(outdir)


    # Generate pseudo labels
    # TODO: add commands for generating pseudo labels

    # Export pseudo labels
    # TODO: add commands for exporting pseudo labels


if __name__ == '__main__':
    main()
