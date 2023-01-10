#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut -c 1-10)

DOCKER_FILE_SHARE=picai_baseline_unet_processor-output-$VOLUME_SUFFIX
docker volume create $DOCKER_FILE_SHARE

docker run  --gpus='"device=2"' --rm \
        -v $SCRIPTPATH/test/:/input/ \
        -v $DOCKER_FILE_SHARE:/output/ \
        picai_baseline_unet_processor

# check detection map (at /output/images/cspca-detection-map/cspca_detection_map.mha)
docker run --rm \
        -v $SCRIPTPATH/test/:/input/ \
        -v $DOCKER_FILE_SHARE:/output/ \
        insighttoolkit/simpleitk-notebooks:latest python -c "import sys; import json; import numpy as np; import SimpleITK as sitk; f1 = sitk.GetArrayFromImage(sitk.ReadImage('/output/images/cspca-detection-map/cspca_detection_map.mha')); f2 = sitk.GetArrayFromImage(sitk.ReadImage('/input/cspca-detection-map/cspca_detection_map.mha')); print('max. difference between prediction and reference:', np.abs(f1-f2).max()); sys.exit(int(np.abs(f1-f2).max() > 1e-3));"

if [ $? -eq 0 ]; then
    echo "Detection map test successfully passed..."
else
    echo "Expected detection map was not found..."
fi

# check case_confidence (at /output/cspca-case-level-likelihood.json)
docker run --rm \
        -v $DOCKER_FILE_SHARE:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        insighttoolkit/simpleitk-notebooks:latest python -c "import sys; import json; f1 = json.load(open('/output/cspca-case-level-likelihood.json')); f2 = json.load(open('/input/cspca-case-level-likelihood.json')); print('Found case-level prediction ' + str(f1) + ', expected ' +str(f2)); sys.exit(int(abs(f1-f2) > 1e-3));"

if [ $? -eq 0 ]; then
    echo "Case-level prediction test successfully passed..."
else
    echo "Expected case-level prediction was not found..."
fi

docker volume rm $DOCKER_FILE_SHARE
