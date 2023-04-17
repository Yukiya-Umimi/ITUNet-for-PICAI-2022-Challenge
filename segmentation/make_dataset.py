import os
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import numpy as np
import h5py
import random

def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()

def csv_reader_single(csv_file,key_col=None,value_col=None):
    '''
    Extracts the specified single column, return a single level dict.
    The value of specified column as the key of dict.

    Args:
    - csv_file: file path
    - key_col: string, specified column as key, the value of the column must be unique. 
    - value_col: string,  specified column as value
    '''
    file_csv = pd.read_csv(csv_file)
    key_list = file_csv[key_col].values.tolist()
    value_list = file_csv[value_col].values.tolist()
    
    target_dict = {}
    for key_item,value_item in zip(key_list,value_list):
        target_dict[key_item] = value_item

    return target_dict

def store_images_labels_2d(save_path, patient_id, cts, labels):

    for i in range(labels.shape[0]):
        ct = cts[:,i,:,:]
        lab = labels[i,:,:]

        hdf5_file = h5py.File(os.path.join(save_path, '%s_%d.hdf5' % (patient_id, i)), 'w')
        hdf5_file.create_dataset('ct', data=ct.astype(np.int16))
        hdf5_file.create_dataset('seg', data=lab.astype(np.uint8))
        hdf5_file.close()


def make_segdata(base_dir,label_dir,output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir_2d = os.path.join(output_dir,'data_2d')
    if not os.path.exists(data_dir_2d):
        os.makedirs(data_dir_2d)
    data_dir_3d = os.path.join(output_dir,'data_3d')
    if not os.path.exists(data_dir_3d):
        os.makedirs(data_dir_3d)

    count = 0

    pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(base_dir)]
    pathlist = list(set(pathlist))
    print(len(pathlist))


    for path in tqdm(pathlist):
        seg = sitk.ReadImage(os.path.join(label_dir,path + '.nii.gz'))

        seg_image = sitk.GetArrayFromImage(seg).astype(np.uint8)
        seg_image[seg_image>=2] = 1
        if np.max(seg_image) == 0:
            continue

        in_1 = sitk.ReadImage(os.path.join(base_dir,path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(base_dir,path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(base_dir,path + '_0002.nii.gz'))

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        img = np.stack((in_1,in_2,in_3),axis=0)

        hdf5_path = os.path.join(data_dir_3d, str(count) + '.hdf5')

        save_as_hdf5(img,hdf5_path,'ct')
        save_as_hdf5(seg_image,hdf5_path,'seg')

        store_images_labels_2d(data_dir_2d,count,img,seg_image)

        count += 1

    print(count)

def make_semidata(base_dir,label_dir,output_dir,test_dir,seg_dir,csv_path):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir_2d = os.path.join(output_dir,'data_2d')
    if not os.path.exists(data_dir_2d):
        os.makedirs(data_dir_2d)
    data_dir_3d = os.path.join(output_dir,'data_3d')
    if not os.path.exists(data_dir_3d):
        os.makedirs(data_dir_3d)

    label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')

    count = 0

    rand_list = list(range(len(label_dict)))
    random.shuffle(rand_list)
    print(rand_list)

    pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(test_dir)]
    pathlist = list(set(pathlist))
    l = len(pathlist)

    for i in tqdm(range(l)):
        path = pathlist[i]

        seg_image = np.load(os.path.join(seg_dir,path + '.npy')).astype(np.uint8)

        seg_image *= int(label_dict[path])

        in_1 = sitk.ReadImage(os.path.join(test_dir,path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(test_dir,path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(test_dir,path + '_0002.nii.gz'))
        

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        img = np.stack((in_1,in_2,in_3),axis=0)

        outc = rand_list[count]

        hdf5_path = os.path.join(data_dir_3d, str(outc) + '.hdf5')

        save_as_hdf5(img,hdf5_path,'ct')
        save_as_hdf5(seg_image,hdf5_path,'seg')

        store_images_labels_2d(data_dir_2d,outc,img,seg_image)

        count += 1

    pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(base_dir)]
    pathlist = list(set(pathlist))


    for path in tqdm(pathlist):
        seg = sitk.ReadImage(os.path.join(label_dir,path + '.nii.gz'))

        seg_image = sitk.GetArrayFromImage(seg).astype(np.uint8)
        seg_image[seg_image==2] = 1
        seg_image[seg_image>2] = 2

        in_1 = sitk.ReadImage(os.path.join(base_dir,path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(base_dir,path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(base_dir,path + '_0002.nii.gz'))

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        img = np.stack((in_1,in_2,in_3),axis=0)

        outc = rand_list[count]


        hdf5_path = os.path.join(data_dir_3d, str(outc) + '.hdf5')

        save_as_hdf5(img,hdf5_path,'ct')
        save_as_hdf5(seg_image,hdf5_path,'seg')

        store_images_labels_2d(data_dir_2d,outc,img,seg_image)

        count += 1


if __name__ == "__main__":
    phase = 'seg'
    base_dir = '../nnUNet_raw_data/Task2201_picai_baseline/imagesTr'
    label_dir = '../nnUNet_raw_data/Task2201_picai_baseline/labelsTr'    
    output_dir = './dataset/segdata'
    test_dir = 'path/to/nnUNet_test_data'
    seg_dir = 'path/to/segmentation_result'
    csv_path = 'path/to/classification_result'
    if phase == 'seg':
        make_segdata(base_dir,label_dir,output_dir)
    else:
        make_semidata(base_dir,label_dir,output_dir,test_dir,seg_dir,csv_path)