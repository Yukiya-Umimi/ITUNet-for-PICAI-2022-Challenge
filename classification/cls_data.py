import os
from pathlib import Path
from typing import Union
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import numpy as np
from skimage.exposure.exposure import rescale_intensity
from PIL import Image
import torch
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import argparse

def get_info(data):
    info = []
    size = list(data.GetSize()[:2])
    z_size = data.GetSize()[-1]
    thick_ness = data.GetSpacing()[-1]
    pixel_spacing = list(data.GetSpacing()[:2])
    info.append(size)
    info.append(z_size)
    info.append(thick_ness)
    info.append(pixel_spacing)
    return info

def get_scale(data):
    info = []
    info.append(np.mean(data))
    info.append(np.max(data))
    info.append(np.min(data))
    info.append(np.std(data))
    return info

def hdf5_reader(data_path, key):
    import h5py
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image

def get_weight_list(ckpt_path,choice=None):
    path_list = []
    for fold in os.scandir(ckpt_path):
        if choice is not None:
            if not (int(fold.name[-1]) in choice or int(fold.name[-5]) in choice):
                continue
        if fold.is_dir():
            weight_path = os.listdir(fold.path)
            weight_path.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            path_list.append(os.path.join(fold.path,weight_path[-1]))
            # print(os.path.join(fold.path,weight_path[-1]))
    return path_list

def store_images_labels_2d(save_path, patient_id, cts, labels):
    plist = []
    llist = []

    for i in range(labels.shape[0]):
        ct = cts[:,i,:,:]
        # ct(3, h, w)
        lab = labels[i,:,:]
        for j in range(ct.shape[0]):
            ct[j] = rescale_intensity(ct[j], out_range=(0, 255))
        img = Image.fromarray(ct.transpose((1,2,0)).astype(np.uint8))

        path = os.path.join(save_path, '%s_%d.png' % (patient_id, i))
        label = np.max(lab) - 1 if np.max(lab)>1 else np.max(lab)
        plist.append(path)
        llist.append(label)

        img.save(path)

    return plist,llist


def make_data(
    base_dir: Union[Path, str] = '../nnUNet_raw_data/Task2201_picai_baseline/imagesTr',
    label_dir: Union[Path, str] = '../nnUNet_raw_data/Task2201_picai_baseline/labelsTr',
    d2_dir: Union[Path, str] = 'path/to/images_illness_3c',
    csv_save_path: Union[Path, str] = 'picai_illness_3c.csv',
):
    os.makedirs(d2_dir, exist_ok=True)

    count = 0

    pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(base_dir)]
    pathlist = list(set(pathlist))
    print(len(pathlist))

    info = {}
    info['id'] = []
    info['label'] = []

    for path in tqdm(pathlist):
        seg = sitk.ReadImage(os.path.join(label_dir,path + '.nii.gz'))

        seg_image = sitk.GetArrayFromImage(seg).astype(np.uint8)
        # seg_image[seg_image>0] = 1
        seg_image[seg_image>3] = 3
        if np.max(seg_image) == 0:
            continue

        in_1 = sitk.ReadImage(os.path.join(base_dir,path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(base_dir,path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(base_dir,path + '_0002.nii.gz'))
        

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        img = np.stack((in_1,in_2,in_3),axis=0)
        # print(img.shape)

        plist,llist = store_images_labels_2d(d2_dir,count,img,seg_image)
        info['id'].extend(plist)
        info['label'].extend(llist)
        count += 1
        # break

    print(count)
    csv_file = pd.DataFrame(info)
    Path(csv_save_path).parent.mkdir(parents=True, exist_ok=True)
    csv_file.to_csv(csv_save_path, index=False)

def predict_test5c(
    weight_path: str = '/opt/cls_algorithm/weights/',
    base_dir: str = 'path/to/nnUNet_test_data',
    csv_save_path: str = 'test_3c.csv',
):
    from efficientnet_pytorch import EfficientNet
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    weight_list = get_weight_list(weight_path,choice=[1,2,3,4,5])
    print(weight_list)
    cls_nets = []
    for weight_path in weight_list:
        cls_net = EfficientNet.from_pretrained(model_name='efficientnet-b5',
                                    in_channels=3,
                                    num_classes=3,
                                    advprop=True)
        checkpoint = torch.load(weight_path)
        cls_net.load_state_dict(checkpoint['state_dict'])
        cls_net.cuda()
        cls_net.eval()
        cls_nets.append(cls_net)

    info = {}
    info['id'] = []
    info['label'] = []

    pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(base_dir)]
    pathlist = list(set(pathlist))
    l = len(pathlist)
    print(l)

    for path in pathlist:
        # count += len(sub_path_list)
        in_1 = sitk.ReadImage(os.path.join(base_dir,path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(base_dir,path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(base_dir,path + '_0002.nii.gz'))
        

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        image = np.stack((in_1,in_2,in_3),axis=0).astype(np.float32)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if np.max(image[i,j]) != 0:
                    image[i,j] = image[i,j]/np.max(image[i,j]) 

        data = torch.from_numpy(image)
        data = data.transpose(1,0).cuda()

        with torch.no_grad():
            cls_results = []
            for cls_net in cls_nets:
                with autocast(True):
                    cls_result = cls_net(data)
                output = F.softmax(cls_result, dim=1)
                # b * c
                output = output.float().squeeze().cpu().numpy()
                cls_results.append(output)
            cls_result = np.mean(np.asarray(cls_results),axis=0)
            cls_result = np.max(cls_result,axis=0)
        # print(cls_result.shape)
        l=np.argmax(cls_result[1:]) + 1 

        print(l)

        info['id'].append(path)
        info['label'].append(l)

    cc= pd.DataFrame(info)
    cc.to_csv(csv_save_path,index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='make_data', choices=["make_data", "predict"],
                        help='choose the mode', type=str)
    args = parser.parse_args()
    if args.mode == "make_data":
        print('makedata')
        make_data()
    else:
        print('predict')
        predict_test5c()