from typing import Tuple
import numpy as np
import skimage as sk
from PIL import Image
from skimage.filters import gaussian
from scipy.ndimage.interpolation import map_coordinates
from imagecorruptions import corrupt
from .util import check_image


class Brightness:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'Brightness@{severity}'
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = corrupt(image1, severity=self.severity, corruption_name='brightness')
        image2 = corrupt(image2, severity=self.severity, corruption_name='brightness')
            
        return image1, image2
    
    
class LowLight:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'LowLight@{severity}'
        
    def lowlight(self, x, severity=1):
        c = [.1, .2, .3, .4, .5][severity - 1]

        x = np.array(x) / 255.

        if len(x.shape) < 3 or x.shape[2] < 3:
            x = np.clip(x - c, 0, 1)
        else:
            x = sk.color.rgb2hsv(x)
            x[:, :, 2] = np.clip(x[:, :, 2] - c, 0, 1)
            x = sk.color.hsv2rgb(x)

        return np.clip(x, 0, 1) * 255
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = self.lowlight(image1, self.severity)
        image2 = self.lowlight(image2, self.severity)
            
        return image1, image2
    
    
class OverExposure:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'OverExposure@{severity}'
        
    def over_exposure(self, x, severity=1):
        ev = [0.4, 0.8, 1.2, 1.6, 2.][severity - 1]
        c = 2 ** ev

        x = np.array(x) / 255.

        if len(x.shape) < 3 or x.shape[2] < 3:
            x = np.clip(x * c, 0, 1)
        else:
            x = sk.color.rgb2hsv(x)
            x[:, :, 2] = np.clip(x[:, :, 2] * c, 0, 1)
            x = sk.color.hsv2rgb(x)

        return np.clip(x, 0, 1) * 255
        
    def exposure(self, x1, x2, severity):
        image_pair = [x1, x2]
        corruption_index = np.random.randint(0, 2)
        image_pair[corruption_index] = self.over_exposure(image_pair[corruption_index], severity)
        
        return image_pair[0], image_pair[1]
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1, image2 = self.exposure(image1, image2, self.severity)
            
        return image1, image2
    
    
class UnderExposure:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'UnderExposure@{severity}'
    
    def under_exposure(self, x, severity=1):
        ev = [-0.4, -0.8, -1.2, -1.6, -2.][severity - 1]
        c = 2 ** ev

        x = np.array(x) / 255.

        if len(x.shape) < 3 or x.shape[2] < 3:
            x = np.clip(x * c, 0, 1)
        else:
            x = sk.color.rgb2hsv(x)
            x[:, :, 2] = np.clip(x[:, :, 2] * c, 0, 1)
            x = sk.color.hsv2rgb(x)

        return np.clip(x, 0, 1) * 255
        
    def exposure(self, x1, x2, severity):
        image_pair = [x1, x2]
        corruption_index = np.random.randint(0, 2)
        image_pair[corruption_index] = self.under_exposure(image_pair[corruption_index], severity)
        
        return image_pair[0], image_pair[1]
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1, image2 = self.exposure(image1, image2, self.severity)
            
        return image1, image2
