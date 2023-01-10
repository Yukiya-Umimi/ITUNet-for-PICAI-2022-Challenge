# Preprocessing Utilities for PICAI 2022 CHALLENGE

We used the preprocess tools provided by Saha et al .

We changed matrix_size in the original processing to [24,384,384] . Then we made the code for generating json files and the code for generating preprocessed data into two python files: make_json.py and make_dataset.py 

You can run them after changing their data path by :

`python make_json.py`
`python make_dataset.py`

Similarly, in the process of preprocessing, we also need to preprocess the unlabeled data, so that we can use the semi supervised method to train our network later. And you should change the date path and output path, too:

```python
    archive_dir="/input/path/to/mha/archive"
    annotations_dir="/input/labels/csPCa_lesion_delineations/human_expert/resampled"
    output_path= "/output/path/to//nnUNet_test_data"
```

Then you could run it by : 

`python precess_testdata.py`

## Reference

● A. Saha, J. J. Twilt, J. S. Bosma, B. van Ginneken, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, "Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge (Study Protocol)", DOI: 10.5281/zenodo.6667655