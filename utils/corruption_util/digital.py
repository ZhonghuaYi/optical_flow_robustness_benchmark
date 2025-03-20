from typing import Tuple
import numpy as np
import skimage as sk
from PIL import Image
from skimage.filters import gaussian
from scipy.ndimage.interpolation import map_coordinates
from imagecorruptions import corrupt
from .util import check_image


class JPEGCompression:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'JPEG@{severity}'
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = corrupt(image1, severity=self.severity, corruption_name='jpeg_compression')
        image2 = corrupt(image2, severity=self.severity, corruption_name='jpeg_compression')
            
        return image1, image2
    

class Pixelation:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'Pixelation@{severity}'
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = corrupt(image1, severity=self.severity, corruption_name='pixelate')
        image2 = corrupt(image2, severity=self.severity, corruption_name='pixelate')
            
        return image1, image2
    

class ElasticTransform:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'ElasticTransform@{severity}'
        
    def elastic_transform(self, x1, x2, severity):
        assert x1.shape == x2.shape
        
        x1 = Image.fromarray(x1)
        x2 = Image.fromarray(x2)
        
        x1 = np.array(x1, dtype=np.float32) / 255.
        x2 = np.array(x2, dtype=np.float32) / 255.
        shape = x1.shape
        shape_size = shape[:2]

        sigma = np.array(shape_size) * 0.01
        alpha = [250 * 0.05, 250 * 0.065, 250 * 0.085, 250 * 0.1, 250 * 0.12][
            severity - 1]
        max_dx = shape[0] * 0.005
        max_dy = shape[0] * 0.005

        dx = (gaussian(np.random.uniform(-max_dx, max_dx, size=shape[:2]),
                    sigma, mode='reflect', truncate=3) * alpha).astype(
            np.float32)
        dy = (gaussian(np.random.uniform(-max_dy, max_dy, size=shape[:2]),
                    sigma, mode='reflect', truncate=3) * alpha).astype(
            np.float32)

        if len(x1.shape) < 3 or x1.shape[2] < 3:
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        else:
            dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
            x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]),
                                np.arange(shape[2]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx,
                                                            (-1, 1)), np.reshape(
                z, (-1, 1))
        
        x1 = np.clip(
            map_coordinates(x1, indices, order=1, mode='reflect').reshape(
                shape), 0, 1) * 255
        x2 = np.clip(
            map_coordinates(x2, indices, order=1, mode='reflect').reshape(
                shape), 0, 1) * 255
        return x1, x2
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1, image2 = self.elastic_transform(image1, image2, self.severity)
            
        return image1, image2
    

class Constrast:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'Contrast@{severity}'
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = corrupt(image1, severity=self.severity, corruption_name='contrast')
        image2 = corrupt(image2, severity=self.severity, corruption_name='contrast')
            
        return image1, image2
    
    
class Saturation:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'Saturation@{severity}'
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = corrupt(image1, severity=self.severity, corruption_name='saturate')
        image2 = corrupt(image2, severity=self.severity, corruption_name='saturate')
            
        return image1, image2
    
    
class LineDropout:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'LineDropout@{severity}'
        
    def line_dropout(self, x, severity=1):
        c = [0.05, 0.10, 0.15, 0.2, 0.25][severity - 1]
        total_lines = x.shape[0]
        total_region_lines = int(total_lines * c)
        
        mask = np.ones_like(x)
        # compute the number of regions
        region_num = np.random.randint(1, min(11, total_region_lines))
        # compute the number of lines in each region
        if region_num == 1:
            region_start = np.random.randint(0, total_lines - total_region_lines + 1)
            mask[region_start:region_start + total_region_lines] = 0
        else:
            region_lines = [0] * region_num
            random_indices = np.sort(np.random.choice(total_region_lines, region_num - 1, replace=False))
            region_lines[0] = random_indices[0]
            for i in range(1, region_num - 1):
                region_lines[i] = random_indices[i] - random_indices[i - 1]
            region_lines[-1] = total_region_lines - random_indices[-1]
            # set the mask
            random_lines = np.sort(np.random.choice(total_lines, total_region_lines, replace=False))
            region_start = random_lines[0]
            mask[region_start:region_start + region_lines[0]] = 0
            j = 0
            for i in range(1, region_num):
                j += region_lines[i - 1]
                region_start = random_lines[j]
                mask[region_start:region_start + region_lines[i]] = 0
        
        # apply the mask
        x = x * mask

        return x
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = self.line_dropout(image1, self.severity)
        image2 = self.line_dropout(image2, self.severity)
            
        return image1, image2
    
