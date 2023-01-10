from src.picai_prep.examples.mha2nnunet.picai_archive import generate_mha2nnunet_settings

generate_mha2nnunet_settings(
    archive_dir="/input/images/",
    annotations_dir="/input/labels/csPCa_lesion_delineations/human_expert/resampled",
    output_path="mha2nnunet_settings2201.json",
    task = "Task2201_picai_baseline"
)

