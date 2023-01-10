from src.picai_prep.preprocessing import Sample,PreprocessingSettings
from src.picai_prep.data_utils import atomic_image_write
import os
import SimpleITK as sitk
from tqdm import tqdm

settings = {
        "preprocessing": {
            # optionally, resample and perform centre crop:
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

def generate_testset(archive_dir,annotations_dir,outdir):
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

            outlist = [os.path.join(outdir,subject_id+ '_' +str(i).zfill(4)+'.nii.gz') for i in range(3)]

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


if __name__ == '__main__':
    archive_dir="/input/path/to/mha/archive"
    annotations_dir="/input/labels/csPCa_lesion_delineations/human_expert/resampled"
    output_path= '/output/path/to/nnUNet_test_data'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    generate_testset(archive_dir,annotations_dir,output_path)