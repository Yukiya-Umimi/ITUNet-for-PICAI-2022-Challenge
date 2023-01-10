# ITUNet-for-PICAI-2022-Challenge

This is the method used by Swangeese Team in PICAI2022 challenge.

This method first uses preprocess to complete the data preprocessing, and the tools are mainly official tools, provided by Saha et al. 
 
Then the classification network and segmentation network are trained respectively. The classification network is EfficientNet-b5, while the segmentation network is ITUNet.
Please note that in the process of generating pseudo labels, the classification network is needed to assist ITUNet to predict. 
 
After training all networks, copy the optimal weights to the weight folder to be submitted, and then rename them. The submitted method still refers to the official method provided by Saha et al., which can be referred to [A](https://github.com/DIAGNijmegen/picai_unet_semi_supervised_gc_algorithm).

## Reference

● Kan H, Shi J, Zhao M, et al. ITUnet: Integration Of Transformers And Unet For Organs-At-Risk Segmentation[C]//2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2022: 2123-2127.

● A. Saha, J. J. Twilt, J. S. Bosma, B. van Ginneken, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, "Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge (Study Protocol)", DOI: 10.5281/zenodo.6667655
