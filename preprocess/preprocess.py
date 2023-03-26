import argparse
import json
import os
from pathlib import Path
from typing import Union

import SimpleITK as sitk
from picai_prep import MHA2nnUNetConverter, atomic_image_write
from picai_prep.examples.mha2nnunet.picai_archive import \
    generate_mha2nnunet_settings
from picai_prep.preprocessing import PreprocessingSettings, Sample
from tqdm import tqdm

from classification.cls_data import make_data
from segmentation.make_dataset import make_segdata

settings = {
    "preprocessing": {
        # resample and perform centre crop:
        "matrix_size": [
            24,
            384,
            384
        ],
        "spacing": [
            3.0,
            0.5,
            0.5
        ],
    }
}


def convert_dataset_unlabeled(
    archive_dir: Union[Path, str],
    annotations_dir: Union[Path, str],
    output_dir: Union[Path, str],
):
    ignore_files = [
        ".DS_Store",
        "LICENSE",
    ]

    for patient_id in tqdm(sorted(os.listdir(archive_dir))):
        # traverse each patient
        if patient_id in ignore_files:
            continue

        # collect list of available studies
        patient_dir = os.path.join(archive_dir, patient_id)
        files = os.listdir(patient_dir)
        files = [fn.replace(".mha", "") for fn in files if ".mha" in fn and "._" not in fn]
        subject_ids = ["_".join(fn.split("_")[0:2]) for fn in files]
        subject_ids = sorted(list(set(subject_ids)))

        # check which studies are complete
        for subject_id in subject_ids:
            patient_id, study_id = subject_id.split("_")

            # construct scan paths
            scan_paths = [
                f"{patient_id}/{subject_id}_{modality}.mha"
                for modality in ["t2w", "adc", "hbv"]
            ]
            all_scans_found = all([
                os.path.exists(os.path.join(archive_dir, path))
                for path in scan_paths
            ])

            if not all_scans_found:
                continue

            outlist = [os.path.join(output_dir,subject_id+ '_' +str(i).zfill(4)+'.nii.gz') for i in range(3)]

            # construct annotation path
            annotation_path = f"{subject_id}.nii.gz"

            if annotations_dir is not None:
                # check if annotation exists
                if os.path.exists(os.path.join(annotations_dir, annotation_path)):
                    # could not find annotation, skip case
                    continue

            # read images
            scans = []
            for scan_properties in scan_paths:
                scans += [sitk.ReadImage(os.path.join(archive_dir,scan_properties))]

            # print(scans[4].GetSize())

            # read label
            lbl = None

            # set up Sample
            sample = Sample(
                scans=scans,
                lbl=lbl,
                settings=PreprocessingSettings(**settings['preprocessing']),
                name=subject_id
            )

            # perform preprocessing
            sample.preprocess()

            # write images
            for scan, scan_properties in zip(sample.scans, outlist):
                atomic_image_write(scan, path=scan_properties, mkdir=True)


def main(taskname="Task2201_picai_baseline"):
    """Preprocess data for (H. Kan et al.)"""
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--workdir', type=str, default="/workdir")
    parser.add_argument('--imagesdir', type=str, default=os.environ.get('SM_CHANNEL_IMAGES', "/input/images"))
    parser.add_argument('--labelsdir', type=str, default=os.environ.get('SM_CHANNEL_LABELS', "/input/picai_labels"))
    parser.add_argument('--outputdir', type=str, default=os.environ.get('SM_MODEL_DIR', "/output"))

    args, _ = parser.parse_known_args()

    # paths
    workdir = Path(args.workdir)
    images_dir = Path(args.imagesdir)
    labels_dir = Path(args.labelsdir)
    output_dir = Path(args.outputdir)
    splits_path = workdir / f"splits/{taskname}/splits.json"
    annotations_dir = labels_dir / "csPCa_lesion_delineations/human_expert/resampled"

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_path.parent.mkdir(parents=True, exist_ok=True)

    # descibe input data
    print(f"workdir: {workdir}")
    print(f"images_dir: {images_dir}")
    print(f"labels_dir: {labels_dir}")
    print(f"output_dir: {output_dir}")

    print("Images folder:", os.listdir(images_dir))
    print("Labels folder:", os.listdir(labels_dir))

    # generate conversion plans
    generate_mha2nnunet_settings(
        archive_dir=images_dir,
        annotations_dir=annotations_dir,
        output_path=splits_path,
        task=taskname,
    )

    # add preprocessing settings to conversion plan
    with open(splits_path, "r") as f:
        mha2nnunet_settings = json.load(f)
    mha2nnunet_settings["preprocessing"] = settings["preprocessing"]

    # convert data to nnU-Net Raw Data format
    archive = MHA2nnUNetConverter(
        scans_dir=images_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir / "nnUNet_raw_data",
        mha2nnunet_settings=mha2nnunet_settings,
    )
    archive.convert()

    # convert dataset without labels for semi-supervised learning
    convert_dataset_unlabeled(
        archive_dir=images_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir / "nnUNet_test_data"
    )

    # save 2D slices for classification
    make_data(
        base_dir=output_dir / "nnUNet_raw_data" / taskname / "imagesTr",
        label_dir=output_dir / "nnUNet_raw_data" / taskname / "labelsTr",
        d2_dir=output_dir / "classification" / "images_illness_3c",
        csv_save_path=output_dir / "classification" / "picai_illness_3c.csv",
    )

    # save 2D slices for segmentation
    make_segdata(
        base_dir=output_dir / "nnUNet_raw_data" / taskname / "imagesTr",
        label_dir=output_dir / "nnUNet_raw_data" / taskname / "labelsTr",
        output_dir=output_dir / "segmentation" / "segdata",
    )


if __name__ == '__main__':
    main()
