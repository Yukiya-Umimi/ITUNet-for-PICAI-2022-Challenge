import sys

sys.path.append('..')

from PIL import Image
from torch.utils.data import Dataset


class DataGenerator(Dataset):
  '''
  Custom Dataset class for data loader.
  Args：
  - path_list: list of file path
  - label_dict: dict, file path as key, label as value
  - transform: the data augmentation methods
  '''
  def __init__(self, path_list, label_dict=None, channels=1, transform=None):

    self.path_list = path_list
    self.label_dict = label_dict
    self.transform = transform
    self.channels = channels


  def __len__(self):
    return len(self.path_list)


  def __getitem__(self,index):
    # Get image and label
    # image: D,H,W
    # label: integer, 0,1,..
    if self.channels == 1:
      image = Image.open(self.path_list[index]).convert('L')
    elif self.channels == 3:
      image = Image.open(self.path_list[index]).convert('RGB')
      
    if self.transform is not None:
      image = self.transform(image)

    if self.label_dict is not None:
      label = self.label_dict[self.path_list[index]] 
      
      sample = {'image':image, 'label':int(label)}
    else:
      sample = {'image':image}
    
    return sample
