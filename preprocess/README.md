# Preprocessing Utilities for PICAI 2022 CHALLENGE

We used the preprocess tools provided by Saha et al.

We changed `matrix_size` in the original preprocessing pipeline to `[24, 384, 384]`. Similarly, in the process of preprocessing, we also need to preprocess the unlabeled data, so that we can use the semi-supervised method to train our network later. The full preprocessing pipeline is captured in `preprocess.py`:

```bash
python preprocess.py \
    --workdir=/workdir \
    --imagesdir=/input/images \
    --labelsdir=/input/picai_labels \
    --outputdir=/output \
    --splits=picai_pub \
```


## Reference

● A. Saha, J. J. Twilt, J. S. Bosma, B. van Ginneken, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, "Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge (Study Protocol)", DOI: 10.5281/zenodo.6667655