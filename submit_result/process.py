#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
from pathlib import Path
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import SimpleITK as sitk
import torch
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from picai_baseline.unet.training_setup.default_hyperparam import \
    get_default_hyperparams
from picai_baseline.unet.training_setup.neural_network_selector import \
    neural_network_for_run
from picai_baseline.unet.training_setup.preprocess_utils import z_score_norm
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import Sample, PreprocessingSettings, crop_or_pad, resample_img
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter
from torch.cuda.amp import autocast as autocast
from model import itunet_2d
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
from scipy import ndimage
import time


class csPCaAlgorithm(SegmentationAlgorithm):
    """
    Wrapper to deploy trained baseline U-Net model from
    https://github.com/DIAGNijmegen/picai_baseline as a
    grand-challenge.org algorithm.
    """

    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # set expected i/o paths in gc env (image i/p, algorithms, prediction o/p)
        # see grand-challenge.org/algorithms/interfaces/ for expected path per i/o interface
        # note: these are fixed paths that should not be modified
        self.start_time = time.time()

        # directory to model weights
        self.algorithm_weights_dir = Path("/opt/algorithm/weights/")
        # self.algorithm_weights_dir = Path("./weights/")

        #path to image files
        self.image_input_dirs = [
            "/input/images/transverse-t2-prostate-mri/",
            "/input/images/transverse-adc-prostate-mri/",
            "/input/images/transverse-hbv-prostate-mri/",
            # "/input/images/coronal-t2-prostate-mri/",  # not used in this algorithm
            # "/input/images/sagittal-t2-prostate-mri/"  # not used in this algorithm
        ]
        # self.image_input_dirs = [
        #     "./test/images/transverse-t2-prostate-mri/",
        #     "./test/images/transverse-adc-prostate-mri/",
        #     "./test/images/transverse-hbv-prostate-mri/",
        #     # "/input/images/coronal-t2-prostate-mri/",  # not used in this algorithm
        #     # "/input/images/sagittal-t2-prostate-mri/"  # not used in this algorithm
        # ]
        self.image_input_paths = [list(Path(x).glob("*.mha"))[0] for x in self.image_input_dirs]

        # load clinical information
        # with open("./test/clinical-information-prostate-mri.json") as fp:
        #     self.clinical_info = json.load(fp)

        # # path to output files
        # self.detection_map_output_path = Path("./test/cspca-detection-map/cspca_detection_map.mha")
        # self.case_level_likelihood_output_file = Path("./test/cspca-case-level-likelihood.json")

        # load clinical information
        with open("/input/clinical-information-prostate-mri.json") as fp:
            self.clinical_info = json.load(fp)

        # path to output files
        self.detection_map_output_path = Path("/output/images/cspca-detection-map/cspca_detection_map.mha")
        self.case_level_likelihood_output_file = Path("/output/cspca-case-level-likelihood.json")

        # create output directory
        self.detection_map_output_path.parent.mkdir(parents=True, exist_ok=True)

        # define compute used for training/inference ('cpu' or 'cuda')
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # define input data specs [image shape, spatial res, num channels, num classes]
        self.img_spec = {
            'image_shape': [24, 384, 384],
            'spacing': [3.0, 0.5, 0.5],
            'num_channels': 3,
            'num_classes': 2,
        }

        # load trained algorithm architecture + weights
        self.models = []
        model_folds = [range(5)]

        # for each given architecture
        for folds in model_folds:

            # for each trained 5-fold instance of a given architecture
            for fold in folds:
                # path to trained weights for this architecture + fold (e.g. 'unet_F4.pt')
                weight_path = self.algorithm_weights_dir / f'F{fold}.pth'

                # skip if model was not trained for this fold
                if not os.path.exists(weight_path):
                    continue

                # define the model specifications used for initialization at train-time
                # note: if the default hyperparam listed in picai_baseline was used,
                # passing arguments 'image_shape', 'num_channels', 'num_classes' and
                # 'model_type' via function 'get_default_hyperparams' is enough.
                # otherwise arguments 'model_strides' and 'model_features' must also
                # be explicitly passed directly to function 'neural_network_for_run'

                model = itunet_2d(n_channels=3,n_classes=3, image_size= tuple([384,384]), transformer_depth = 24)

                # load trained weights for the fold
                checkpoint = torch.load(weight_path,map_location=self.device)
                model.load_state_dict(checkpoint['state_dict'])
                model.to(self.device)
                self.models += [model]
                print("Complete.")
                print("-"*100)

        # path to trained weights for this architecture + fold (e.g. 'unet_F4.pt')
        self.cls_models = []
        model_folds = [range(5)]
        # weight_path = self.algorithm_weights_dir / 'CLS_F0.pth'
        for folds in model_folds:
    
            # for each trained 5-fold instance of a given architecture
            for fold in folds:
                # path to trained weights for this architecture + fold (e.g. 'unet_F4.pt')
                weight_path = self.algorithm_weights_dir / f'CLS_F{fold}.pth'

                # skip if model was not trained for this fold
                if not os.path.exists(weight_path):
                    continue

                model = EfficientNet.from_name(model_name='efficientnet-b5')
                num_ftrs = model._fc.in_features
                model._fc = torch.nn.Linear(num_ftrs, 3)
                # load trained weights for the fold
                checkpoint = torch.load(weight_path,map_location=self.device)
                model.load_state_dict(checkpoint['state_dict'])
                model.to(self.device)
                self.cls_models += [model]
        print("Complete.")
        print("-"*100)

        # display error/success message
        if len(self.models) == 0:
            raise Exception("No models have been found/initialized.")
        else:
            print(f"Success! {len(self.models)} model(s) have been initialized.")
            print("-"*100)

    # generate + save predictions, given images
    def predict(self):

        print("Preprocessing Images ...")

        # read images (axial sequences used for this example only)
        sample = Sample(
            scans=[
                sitk.ReadImage(str(path))
                for path in self.image_input_paths
            ],
            settings=PreprocessingSettings(
                matrix_size=self.img_spec['image_shape'], 
                spacing=self.img_spec['spacing']
            )
        )

        # preprocess - align, center-crop, resample
        sample.preprocess()
        cropped_img = [
            sitk.GetArrayFromImage(x).astype(np.int16)
            for x in sample.scans
        ]
        image = np.stack(cropped_img,axis=0).astype(np.float32)

        zero_mask = np.ones((24,),dtype=np.float32)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if np.max(image[i,j]) != 0:
                    image[i,j] = image[i,j]/np.max(image[i,j]) 
                else:
                    zero_mask[i] = 0

        data = torch.from_numpy(image)
        data = data.transpose(1,0).to(self.device)

        cls_results = []

        for p in range(len(self.cls_models)):

            self.cls_models[p].eval()
            with torch.no_grad():
                with autocast(True):
                    cls_result = self.cls_models[p](data)
            cls_result = F.softmax(cls_result.float(), dim=1)
            cls_result = cls_result[:,1].detach().cpu().numpy()
            cls_results.append(cls_result)

        cls_result = np.mean(np.asarray(cls_results),axis=0)
        cls_result = cls_result * zero_mask

        cls_result.sort()
        # print(cls_result)
        cls_p = np.mean(cls_result[-7:])

        outputs = []
        print("Generating Predictions ...")

        # for each member model in ensemble
        for p in range(len(self.models)):

            # switch model to evaluation mode
            self.models[p].eval()

            # scope to disable gradient updates
            with torch.no_grad():
                rs = []
                for i in range(data.size()[0]):
                    with autocast(False):
                        output = self.models[p](data[i:i+1,...])
                    if isinstance(output,tuple) or isinstance(output,list):
                        output = output[0]
                    else:
                        output = output  
                    # print(seg_output.size())
                    output = output.float()
                    output = torch.softmax(output,dim=1).squeeze().detach().cpu().numpy()  #N*H*W 
                    output = output[0]
                    rs.append(output)

                output = np.stack(rs,axis=0)
                # print(output.shape)

                # gaussian blur to counteract checkerboard artifacts in
                # predictions from the use of transposed conv. in the U-Net
                outputs += [
                    output
                ]

        # ensemble softmax predictions
        ensemble_output = np.mean(outputs, axis=0).astype('float32')
        ensemble_output = 1 - ensemble_output

        print("ensemble_output OK!!!")

        # read and resample images (used for reverting predictions only)
        sitk_img = [
            sitk.ReadImage(str(path)) for path in self.image_input_paths
        ]
        resamp_img = [
            sitk.GetArrayFromImage(
                resample_img(x, out_spacing=self.img_spec['spacing'])
            )
            for x in sitk_img
        ]

        # revert softmax prediction to original t2w - reverse center crop
        cspca_det_map_sitk: sitk.Image = sitk.GetImageFromArray(crop_or_pad(
            ensemble_output, size=resamp_img[0].shape))
        cspca_det_map_sitk.SetSpacing(list(reversed(self.img_spec['spacing'])))

        # revert softmax prediction to original t2w - reverse resampling
        cspca_det_map_sitk = resample_img(cspca_det_map_sitk,
                                          out_spacing=list(reversed(sitk_img[0].GetSpacing())))

        # process softmax prediction to detection map
        cspca_det_map_npy = extract_lesion_candidates(
            sitk.GetArrayFromImage(cspca_det_map_sitk), threshold='dynamic')[0]

        # remove (some) secondary concentric/ring detections
        cspca_det_map_npy[cspca_det_map_npy<(np.max(cspca_det_map_npy)/2)] = 0

        # make sure that expected shape was matched after reverse resampling (can deviate due to rounding errors)
        cspca_det_map_npy = crop_or_pad(
            cspca_det_map_npy, size=sitk.GetArrayFromImage(sitk_img[0]).shape)
        cspca_det_map_sitk: sitk.Image = sitk.GetImageFromArray(cspca_det_map_npy)
        # print(cspca_det_map_sitk.GetSize())

        # works only if the expected shape matches
        cspca_det_map_sitk.CopyInformation(sitk_img[0])

        # save detection map
        atomic_image_write(cspca_det_map_sitk, self.detection_map_output_path)
        print('cspca_det_map_sitk write OK!!')

        # save case-level likelihood
        with open(str(self.case_level_likelihood_output_file), 'w') as f:
            # json.dump((float(np.max(cspca_det_map_npy))+float(cls_p))/2, f)
            json.dump(float(np.max(cspca_det_map_npy)), f)
        # print(np.max(cspca_det_map_npy))
        print('finished!!')

if __name__ == "__main__":
    csPCaAlgorithm().predict()
