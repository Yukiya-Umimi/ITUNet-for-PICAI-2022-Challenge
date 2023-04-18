import argparse
import os
from pathlib import Path

from classification.cls_data import predict_test5c
from segmentation.make_dataset import make_semidata
from segmentation.predict_2d import Config, postprecess, save_npy, vote_dir


def main():
    """Prepare detection dataset for method from Swangeese team for PI-CAI Challenge."""
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--workdir', type=str, default="/workdir")
    parser.add_argument('--preprocesseddir', type=str, default=os.environ.get('SM_CHANNEL_PREPROCESSED', "/input/preprocessed"))
    parser.add_argument('--supervisedweightsdir', type=str, default=os.environ.get('SM_CHANNEL_SUPERVISED_WEIGHTS', "/input/weights"))
    parser.add_argument('--outputdir', type=str, default=os.environ.get('SM_MODEL_DIR', "/output"))

    args, _ = parser.parse_known_args()

    # paths
    workdir = Path(args.workdir)
    output_dir = Path(args.outputdir)
    preprocessed_dir = Path(args.preprocesseddir)
    supervised_weights_dir = Path(args.supervisedweightsdir)

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # descibe input data
    print(f"workdir: {workdir}")
    print(f"output_dir: {output_dir}")

    # Perform inference with the classification model
    predict_test5c(
        weight_path=supervised_weights_dir / 'ckpt/picai/v0/',
        base_dir=preprocessed_dir / 'nnUNet_test_data',
        csv_save_path=workdir / 'test_3c.csv',
    )

    # Perform inference with the segmentation model
    data_path = preprocessed_dir / "nnUNet_test_data"
    outdir = workdir / 'segout/segmentation_result'
    save_npy(
        data_path=data_path,
        ckpt_path_base=supervised_weights_dir / 'ckpt/seg',
        save_dir_base=workdir / 'segout',
    )

    # Ensemble segmentation results
    config = Config()
    vote_dir(datadir=workdir / f'segout/{config.version}')

    # Postprocess segmentation results
    postprecess(
        data_dir=workdir / f'segout/{config.version}/avg',
        outdir=outdir,
    )

    # Prepare dataset for detection
    make_semidata(
        base_dir=preprocessed_dir / 'nnUNet_raw_data/Task2201_picai_baseline/imagesTr',
        label_dir=preprocessed_dir / 'nnUNet_raw_data/Task2201_picai_baseline/labelsTr',
        output_dir=output_dir,
        test_dir=preprocessed_dir / "nnUNet_test_data",
        seg_dir=outdir,
        csv_path=workdir / 'test_3c.csv',
    )


if __name__ == '__main__':
    main()
