from src.picai_prep import MHA2nnUNetConverter

archive = MHA2nnUNetConverter(
    scans_dir="/input/path/to/mha/archive",
    annotations_dir="/input/path/to/annotations",  # defaults to input_path
    output_dir="/output/path/to/nnUNet_raw_data",
    mha2nnunet_settings="mha2nnunet_settings2201.json",
)
archive.convert()