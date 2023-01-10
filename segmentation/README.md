# Segmentation Part for PICAI 2022 CHALLENGE

## Perpare Segmentation Data
Using the make_dataset.py to prepare the training data of segmentation phase.
You should change the path to your data path, and make sure the 'phase' should be set to 'seg'
Then you could run
`python make_dataset.py`

After that, you will find that the training data is structured in the following way:

```
/path/to/dataset/
├── segdata/
    ├── data_2d/
        ├── [patient UID]_[piece UID].hdf5
        ...
    ├── data_3d/
        ├── [patient UID].hdf5
        ...
```

## Training Supervised Phase
Before trainint ITUNet, you should check the training config is suitable for your device, 
you could set it in config.py, and make sure that "PHASE" should be set to 'seg'.

 As shown below :
 ```python
PHASE = 'seg'  
DEVICE = '1,2,3' #using GPUs
PATH_DIR = './dataset/segdata/data_2d'  #segdata dir
PATH_AP = './dataset/segdata/data_3d'
```

And then, you could run 
`python run.py -m train-cross`
for training 5 folds.

## Generate Pseudo Label
After you get the weights of the segmentation network, you can use them to predict the pseudo labels for unlabeled data.
Set the data path in predict_2d.py correctly, and then run it by:
`python predict_2d.py`

And then you could find your result in your output dir.

## Perpare Detection Data
Using the make_dataset.py to prepare the training data of detection phase.
You should change the path to your data path, and make sure the 'phase' should be set to 'detect', and the output_dir should be different from before.

Please note that the data generation of this part also requires the prediction results of the classification network.
You could set it as below:
 ```python
    phase = 'detect'
    base_dir = '../nnUNet_raw_data/Task2201_picai_baseline/imagesTr'
    label_dir = '../nnUNet_raw_data/Task2201_picai_baseline/labelsTr'    
    output_dir = './dataset/detectdata'
    test_dir = 'path/to/nnUNet_test_data'
    seg_dir = 'path/to/segmentation_result'
    csv_path = 'path/to/classification_result'
```

Then you could run
`python make_dataset.py`

After that, you will find that the training data is structured in the following way:

```
/path/to/dataset/
├── detectdata/
    ├── data_2d/
        ├── [patient UID]_[piece UID].hdf5
        ...
    ├── data_3d/
        ├── [patient UID].hdf5
        ...
```

## Training Detection Phase
Before trainint ITUNet, you should check the training config is suitable for your device, 
you could set it in config.py, and make sure that "PHASE" should be set to 'detect'.

 As shown below :
 ```python
PHASE = 'detect'  
DEVICE = '1,2,3' #using GPUs
PATH_DIR = './dataset/detectdata/data_2d'  #segdata dir
PATH_AP = './dataset/detectdata/data_3d'
```

And then, you could run 
`python run.py -m train-cross`
for training 5 folds.

 At this phase, we used the tools provided by saha et al. to evaluate the effect of the network on the verification set, and retained the best weight.

After the training, the weights you get can be used to submit. 

## Reference

● Kan H, Shi J, Zhao M, et al. ITUnet: Integration Of Transformers And Unet For Organs-At-Risk Segmentation[C]//2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2022: 2123-2127.

● A. Saha, J. J. Twilt, J. S. Bosma, B. van Ginneken, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, "Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge (Study Protocol)", DOI: 10.5281/zenodo.6667655