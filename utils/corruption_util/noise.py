from typing import Tuple
import numpy as np
from imagecorruptions import corrupt
from .util import check_image


class GaussianNoise:
    def __init__(self, severity=5):
        self.severity = severity
        self.name = f'GaussianNoise@{severity}'

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = corrupt(image1, severity=self.severity, corruption_name='gaussian_noise')
        image2 = corrupt(image2, severity=self.severity, corruption_name='gaussian_noise')
            
        return image1, image2


class ShotNoise:
    def __init__(self, severity=5):
        self.severity = severity
        self.name = f'ShotNoise@{severity}'

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = corrupt(image1, severity=self.severity, corruption_name='shot_noise')
        image2 = corrupt(image2, severity=self.severity, corruption_name='shot_noise')
            
        return image1, image2
    

class ImpulseNoise:
    def __init__(self, severity=5):
        self.severity = severity
        self.name = f'ImpulseNoise@{severity}'

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = corrupt(image1, severity=self.severity, corruption_name='impulse_noise')
        image2 = corrupt(image2, severity=self.severity, corruption_name='impulse_noise')
            
        return image1, image2
    

class SpeckleNoise:
    def __init__(self, severity=5):
        self.severity = severity
        self.name = f'SpeckleNoise@{severity}'

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = corrupt(image1, severity=self.severity, corruption_name='speckle_noise')
        image2 = corrupt(image2, severity=self.severity, corruption_name='speckle_noise')
            
        return image1, image2
