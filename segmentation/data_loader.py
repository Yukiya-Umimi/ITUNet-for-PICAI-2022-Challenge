import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from segmentation.utils import hdf5_reader


class Normalize(object):
  def __call__(self,sample):
    ct = sample['ct']
    seg = sample['seg']
    for i in range(ct.shape[0]):
        if np.max(ct[i])!=0:
            ct[i] = ct[i]/np.max(ct[i])
        
    ct[ct<0] = 0

    new_sample = {'ct':ct, 'seg':seg}
    return new_sample

class RandomRotate2D(object):
    """
    Data augmentation method.
    Rotating the image with random degree.
    Args:
    - degree: the rotate degree from (-degree , degree)
    Returns:
    - rotated image and label
    """

    def __init__(self, degree=[-15,-10,-5,0,5,10,15]):
        self.degree = degree

    def __call__(self, sample):
        ct_image = sample['ct']
        label = sample['seg']

        cts = []
        for i in range(ct_image.shape[0]):
            cts.append(Image.fromarray(ct_image[i]))
        label = Image.fromarray(np.uint8(label))

        rotate_degree = random.choice(self.degree)

        cts_out = []
        for ct in cts:
            ct = ct.rotate(rotate_degree, Image.BILINEAR)
            ct = np.array(ct).astype(np.float32)
            cts_out.append(ct)

        label = label.rotate(rotate_degree, Image.NEAREST)

        ct_image = np.asarray(cts_out)
        label = np.array(label).astype(np.float32)
        return {'ct':ct_image, 'seg': label}

class RandomFlip2D(object):
    '''
    Data augmentation method.
    Flipping the image, including horizontal and vertical flipping.
    Args:
    - mode: string, consisting of 'h' and 'v'. Optional methods and 'hv' is default.
            'h'-> horizontal flipping,
            'v'-> vertical flipping,
            'hv'-> random flipping.
    '''
    def __init__(self, mode='hv'):
        self.mode = mode

    def __call__(self, sample):
        ct_image = sample['ct']
        label = sample['seg']

        if 'h' in self.mode and 'v' in self.mode:
            random_factor = np.random.uniform(0, 1)
            if random_factor < 0.3:
                ct_image = ct_image[:,:,::-1]
                label = label[:,::-1]
            elif random_factor < 0.6:
                ct_image = ct_image[:,::-1,:]
                label = label[::-1,:]

        elif 'h' in self.mode:
            if np.random.uniform(0, 1) > 0.5:
                ct_image = ct_image[:,:,::-1]
                label = label[:,::-1]

        elif 'v' in self.mode:
            if np.random.uniform(0, 1) > 0.5:
                ct_image = ct_image[:,::-1,:]
                label = label[::-1,:]

        ct_image = ct_image.copy()
        label = label.copy()
        return {'ct':ct_image, 'seg': label}

class To_Tensor(object):
  '''
  Convert the data in sample to torch Tensor.
  Args:
  - n_class: the number of class
  '''
  def __init__(self,num_class=2,input_channel = 3):
    self.num_class = num_class
    self.channel = input_channel

  def __call__(self,sample):

    ct = sample['ct']
    seg = sample['seg']

    new_image = ct[:self.channel,...]
    new_label = np.empty((self.num_class,) + seg.shape, dtype=np.float32)
    for z in range(1, self.num_class):
        temp = (seg==z).astype(np.float32)
        new_label[z,...] = temp
    new_label[0,...] = np.amax(new_label[1:,...],axis=0) == 0   
   
    # convert to Tensor
    new_sample = {'image': torch.from_numpy(new_image),
                  'label': torch.from_numpy(new_label)}
    
    return new_sample

class DataGenerator(Dataset):
  '''
  Custom Dataset class for data loader.
  Args：
  - path_list: list of file path
  - roi_number: integer or None, to extract the corresponding label
  - num_class: the number of classes of the label
  - transform: the data augmentation methods
  '''
  def __init__(self, path_list, num_class=2, transform=None):

    self.path_list = path_list
    self.num_class = num_class
    self.transform = transform


  def __len__(self):
    return len(self.path_list)


  def __getitem__(self,index):

    ct = hdf5_reader(self.path_list[index],'ct')
    seg = hdf5_reader(self.path_list[index],'seg') 

    sample = {'ct': ct, 'seg':seg}
    # Transform
    if self.transform is not None:
      sample = self.transform(sample)

    return sample

