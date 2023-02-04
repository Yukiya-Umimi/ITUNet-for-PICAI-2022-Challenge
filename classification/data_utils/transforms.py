import torchvision.transforms.functional as TF
import random

class RandomRotate(object):

    def __init__(self, angels):
        self.angels = angels

    def __call__(self, image):
        
        angle = random.choice(self.angels)
        image = TF.rotate(image, angle)
        return image
