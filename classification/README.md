# Classification Part for PICAI 2022 CHALLENGE

## Perpare Classification Data
Using the cls_data.py to prepare the training data of Classification phase.
You should change the path to your data path
Then you could run
`python cls_data.py -m make_data`

After that, you will find that the training data is structured in the following way:

```
/path/to/images_illness_3c/
├── images_illness_3c/
    ├── [patient UID]_[piece UID].png
    ...
```

## Training Phase
Before training Efficientnet-b5, you should check the training config is suitable for your device, 
you could set it in config.py.

 As shown below :
 ```python
CSV_PATH = './picai_illness_3c.csv'  
DEVICE = '0' #using GPUs
```

And then, you could run 
`python run.py -m train-cross`
for training 5 folds.

## Generate Pseudo Label for Segmentation
After you get the weights of the classification network, you can use them to predict the pseudo labels for unlabeled data.
Set the data path in cls_data.py correctly, and then run it by:
`python cls_data.py -m predict`

And then you could find your result in your output dir.


## Reference

● Kan H, Shi J, Zhao M, et al. ITUnet: Integration Of Transformers And Unet For Organs-At-Risk Segmentation[C]//2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2022: 2123-2127.

● A. Saha, J. J. Twilt, J. S. Bosma, B. van Ginneken, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, "Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge (Study Protocol)", DOI: 10.5281/zenodo.6667655