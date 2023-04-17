import os

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from torch.cuda.amp import autocast as autocast

from segmentation.model import itunet_2d
from segmentation.utils import get_weight_path


def predict_process(test_path,config,base_dir):
    # get weight
    weight_path = get_weight_path(config.ckpt_path)
    print(weight_path)

    # get net
    net = itunet_2d(n_channels=config.channels,n_classes=config.num_classes, image_size= tuple((384,384)), transformer_depth = 24)
    checkpoint = torch.load(weight_path)
    net.load_state_dict(checkpoint['state_dict'])

    pred = []
    net = net.cuda()
    net.eval()

    in_1 = sitk.ReadImage(os.path.join(base_dir,test_path + '_0000.nii.gz'))
    in_2 = sitk.ReadImage(os.path.join(base_dir,test_path + '_0001.nii.gz'))
    in_3 = sitk.ReadImage(os.path.join(base_dir,test_path + '_0002.nii.gz'))

    in_1 = sitk.GetArrayFromImage(in_1).astype(np.float32)
    in_2 = sitk.GetArrayFromImage(in_2).astype(np.float32)
    in_3 = sitk.GetArrayFromImage(in_3).astype(np.float32)

    image = np.stack((in_1,in_2,in_3),axis=0)

    with torch.no_grad():
        for i in range(image.shape[1]):
            new_image = image[:,i,:,:]
            for j in range(new_image.shape[0]):
                if np.max(new_image[j]) != 0:
                    new_image[j] = new_image[j]/np.max(new_image[j])
            new_image = np.expand_dims(new_image,axis=0)
            data = torch.from_numpy(new_image)

            data = data.cuda()
            with autocast(False):
                output = net(data)
            if isinstance(output,tuple) or isinstance(output,list):
                seg_output = output[0]
            else:
                seg_output = output  
            seg_output = torch.softmax(seg_output, dim=1).detach().cpu().numpy()   
            pred.append(seg_output) 
    
    pred = np.concatenate(pred,axis=0).transpose((1,0,2,3))
    return pred

def save_npy(data_path):
    config = Config()
    for fold in range(1,6):
        print('****fold%d****'%fold)
        config.fold = fold
        config.ckpt_path = f'./new_ckpt/seg/{config.version}/fold{str(fold)}'
        save_dir = f'./segout/{config.version}/fold{str(fold)}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(data_path)]
        pathlist = list(set(pathlist))

        for path in pathlist:
            pred = predict_process(path,config,data_path)
            print(pred.shape)
            np.save(os.path.join(save_dir,path+'.npy'),pred)

def vote_dir(datadir = None):
    config = Config()
    if datadir is None:
        datadir = f'./segout/{config.version}'
    outdir = os.path.join(datadir,'avg')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    path_list = list(os.listdir(os.path.join(datadir,'fold1')))
    for path in path_list:
        re  = np.stack([np.load(os.path.join(datadir,'fold'+str(i),path))for i in range(1,6)],axis=0)
        re = np.mean(re,axis=0)
        np.save(os.path.join(outdir,path),re)

def postprecess(outdir):
    config = Config()
    data_dir = f'./segout/{config.version}/avg'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    path_list = list(os.listdir(data_dir))
    label_structure = np.ones((3, 3, 3))
    for path in path_list:
        temp = np.load(os.path.join(data_dir,path))
        temp = temp[1]
        temp[temp<0.5] = 0
        from report_guided_annotation import extract_lesion_candidates

        # process softmax prediction to detection map
        cspca_det_map_npy = extract_lesion_candidates(
            temp, threshold='dynamic')[0]

        # remove (some) secondary concentric/ring detections
        cspca_det_map_npy[cspca_det_map_npy<(np.max(cspca_det_map_npy)/2)] = 0

        blobs_index, num_blobs = ndimage.label(cspca_det_map_npy, structure=label_structure)
        max_b,max_s = 0,0
        temp = np.zeros(cspca_det_map_npy.shape,dtype=np.uint8)
        for lesion_candidate_id in range(num_blobs):
            s = np.sum(blobs_index == (1+lesion_candidate_id))
            if s > max_s:
                max_s = s
                max_b = lesion_candidate_id
        for lesion_candidate_id in range(num_blobs):
            if lesion_candidate_id != max_b and np.sum(cspca_det_map_npy[blobs_index == (1+lesion_candidate_id)]) <= 1000:
                cspca_det_map_npy[blobs_index == (1+lesion_candidate_id)] = 0

        blobs_index, num_blobs = ndimage.label(cspca_det_map_npy, structure=label_structure)
        
        temp[cspca_det_map_npy>0.5] = 1
        print(temp.shape,temp.dtype)
        print(np.sum(temp),num_blobs)
        np.save(os.path.join(outdir,path),temp)
    print(len(path_list))
            
class Config:    
    input_shape = (384,384)
    channels = 3
    num_classes = 2

    version = 'itunet_d24'
    fold = 1
    ckpt_path = f'./new_ckpt/seg/{version}/fold{str(fold)}'

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # test data
    data_path = '/staff/honeyk/project/picai_prep-main/open_source/nnUNet_test_data'
    outdir = './segout/segmentation_result'
    save_npy(data_path)
    vote_dir()
    postprecess(outdir)
