call .\build.bat

docker volume create picai_baseline_unet_processor-output

docker run --rm --gpus all^
        --memory=16g --memory-swap=16g^
        --cap-drop=ALL --security-opt="no-new-privileges"^
        --network none --shm-size=128m --pids-limit 256^
        -v %~dp0\test\:/input/^
        -v picai_baseline_unet_processor-output:/output/^
        picai_baseline_unet_processor

:: check detection map (at /output/images/cspca-detection-map/cspca_detection_map.mha)
docker run --rm^
        -v picai_baseline_unet_processor-output:/output/^
        -v %~dp0\test\:/input/^
        insighttoolkit/simpleitk-notebooks:latest python -c "import sys; import json; import numpy as np; import SimpleITK as sitk; f1 = sitk.GetArrayFromImage(sitk.ReadImage('/output/images/cspca-detection-map/cspca_detection_map.mha')); f2 = sitk.GetArrayFromImage(sitk.ReadImage('/input/cspca-detection-map/cspca_detection_map.mha')); print('max. difference between prediction and reference:', np.abs(f1-f2).max()); sys.exit(int(np.abs(f1-f2).max() > 1e-3));"

if %ERRORLEVEL% == 0 (
    echo "Detection map test successfully passed..."
) else (
    echo "Expected detection map was not found..."
)

:: check case_confidence (at /output/cspca-case-level-likelihood.json)
docker run --rm^
        -v picai_baseline_unet_processor-output:/output/^
        -v %~dp0\test\:/input/^
        insighttoolkit/simpleitk-notebooks:latest python -c "import sys; import json; f1 = json.load(open('/output/cspca-case-level-likelihood.json')); f2 = json.load(open('/input/cspca-case-level-likelihood.json')); print('Found case-level prediction ' + str(f1) + ', expected ' +str(f2)); sys.exit(int(abs(f1-f2) > 1e-3));"

if %ERRORLEVEL% == 0 (
    echo "Case-level prediction test successfully passed..."
) else (
    echo "Expected case-level prediction was not found..."
)

:: cleanup
docker volume rm picai_baseline_unet_processor-output