from typing import Tuple
import os
import numpy as np
from PIL import Image
import cv2 as cv
from imagecorruptions import corrupt
from skimage.filters import gaussian
from .util import check_image, next_power_of_2, plasma_fractal, rgb2gray, clipped_zoom, _motion_blur


class Spatter:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'Spatter@{severity}'
        
    def spatter(self, x1, x2, severity):
        assert x1.shape == x2.shape
        
        x1 = Image.fromarray(x1)
        x2 = Image.fromarray(x2)
        
        c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
            (0.65, 0.3, 3, 0.68, 0.6, 0),
            (0.65, 0.3, 2, 0.68, 0.5, 0),
            (0.65, 0.3, 1, 0.65, 1.5, 1),
            (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
        x1_PIL = x1
        x2_PIL = x2
        x1 = np.array(x1, dtype=np.float32) / 255.
        x2 = np.array(x2, dtype=np.float32) / 255.

        liquid_layer = np.random.normal(size=x1.shape[:2], loc=c[0], scale=c[1])

        liquid_layer = gaussian(liquid_layer, sigma=c[2])
        liquid_layer[liquid_layer < c[3]] = 0
        if c[5] == 0:
            liquid_layer = (liquid_layer * 255).astype(np.uint8)
            dist = 255 - cv.Canny(liquid_layer, 50, 150)
            dist = cv.distanceTransform(dist, cv.DIST_L2, 5)
            _, dist = cv.threshold(dist, 20, 20, cv.THRESH_TRUNC)
            dist = cv.blur(dist, (3, 3)).astype(np.uint8)
            dist = cv.equalizeHist(dist)
            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = cv.filter2D(dist, cv.CV_8U, ker)
            dist = cv.blur(dist, (3, 3)).astype(np.float32)

            m = cv.cvtColor(liquid_layer * dist, cv.COLOR_GRAY2BGRA)
            m /= np.max(m, axis=(0, 1))
            m *= c[4]
            # water is pale turqouise
            color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1])), axis=2)

            color = cv.cvtColor(color, cv.COLOR_BGR2BGRA)

            if len(x1.shape) < 3 or x1.shape[2] < 3:
                add_spatter_color = cv.cvtColor(np.clip(m * color, 0, 1),
                                                cv.COLOR_BGRA2BGR)
                add_spatter_gray = rgb2gray(add_spatter_color)
                
                x1 = np.clip(x1 + add_spatter_gray, 0, 1) * 255
                x2 = np.clip(x2 + add_spatter_gray, 0, 1) * 255

                return x1, x2

            else:

                x1 = cv.cvtColor(x1, cv.COLOR_BGR2BGRA)
                x2 = cv.cvtColor(x2, cv.COLOR_BGR2BGRA)
                
                x1 = cv.cvtColor(np.clip(x1 + m * color, 0, 1), cv.COLOR_BGRA2BGR) * 255
                x2 = cv.cvtColor(np.clip(x2 + m * color, 0, 1), cv.COLOR_BGRA2BGR) * 255

                return x1, x2
        else:
            m = np.where(liquid_layer > c[3], 1, 0)
            m = gaussian(m.astype(np.float32), sigma=c[4])
            m[m < 0.8] = 0

            x1_rgb = np.array(x1_PIL.convert('RGB'))
            x2_rgb = np.array(x2_PIL.convert('RGB'))

            # mud brown
            color = np.concatenate((63 / 255. * np.ones_like(x1_rgb[..., :1]),
                                    42 / 255. * np.ones_like(x1_rgb[..., :1]),
                                    20 / 255. * np.ones_like(x1_rgb[..., :1])),
                                axis=2)
            color *= m[..., np.newaxis]
            if len(x1.shape) < 3 or x1.shape[2] < 3:
                x1 *= (1 - m)
                x1 = np.clip(x1 + rgb2gray(color), 0, 1) * 255
                x2 *= (1 - m)
                x2 = np.clip(x2 + rgb2gray(color), 0, 1) * 255
                return x1, x2

            else:
                x1 *= (1 - m[..., np.newaxis])
                x2 *= (1 - m[..., np.newaxis])
                x1 = np.clip(x1 + color, 0, 1) * 255
                x2 = np.clip(x2 + color, 0, 1) * 255
                return x1, x2
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1, image2 = self.spatter(image1, image2, self.severity)
            
        return image1, image2
    

class Fog:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'Fog@{severity}'
        
    def fog(self, x1, x2, severity):
        assert x1.shape == x2.shape
        
        x1 = Image.fromarray(x1)
        x2 = Image.fromarray(x2)
        
        c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
        
        shape = np.array(x1).shape
        max_side = np.max(shape)
        map_size = next_power_of_2(int(max_side))

        x1 = np.array(x1) / 255.
        max_val1 = x1.max()
        x2 = np.array(x2) / 255.
        max_val2 = x2.max()

        x_shape = np.array(x1).shape
        if len(x_shape) < 3 or x_shape[2] < 3:
            fog_map = plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0], :shape[1]]
            x1 += c[0] * fog_map
            x2 += c[0] * fog_map
        else:
            fog_map = plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0], :shape[1]][..., np.newaxis]
            x1 += c[0] * fog_map
            x2 += c[0] * fog_map
        
        x1 = np.clip(x1 * max_val1 / (max_val1 + c[0]), 0, 1) * 255
        x2 = np.clip(x2 * max_val2 / (max_val2 + c[0]), 0, 1) * 255
        return x1, x2
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1, image2 = self.fog(image1, image2, self.severity)
            
        return image1, image2
    
    
class Frost:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'Frost@{severity}'
        
    def frost(self, x1, x2, severity):
        assert x1.shape == x2.shape
        
        x1 = Image.fromarray(x1)
        x2 = Image.fromarray(x2)
        
        c = [(1, 0.4),
            (0.8, 0.6),
            (0.7, 0.7),
            (0.65, 0.7),
            (0.6, 0.75)][severity - 1]

        idx = np.random.randint(5)
        
        filename = ['frost/frost1.png',
                    'frost/frost2.png',
                    'frost/frost3.png',
                    'frost/frost4.jpg',
                    'frost/frost5.jpg',
                    'frost/frost6.jpg',][idx]
        current_path = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(current_path, filename)
        frost = cv.imread(filename)
        frost_shape = frost.shape
        x_shape = np.array(x1).shape

        # resize the frost image so it fits to the image dimensions
        scaling_factor = 1
        if frost_shape[0] >= x_shape[0] and frost_shape[1] >= x_shape[1]:
            scaling_factor = 1
        elif frost_shape[0] < x_shape[0] and frost_shape[1] >= x_shape[1]:
            scaling_factor = x_shape[0] / frost_shape[0]
        elif frost_shape[0] >= x_shape[0] and frost_shape[1] < x_shape[1]:
            scaling_factor = x_shape[1] / frost_shape[1]
        elif frost_shape[0] < x_shape[0] and frost_shape[1] < x_shape[
            1]:  # If both dims are too small, pick the bigger scaling factor
            scaling_factor_0 = x_shape[0] / frost_shape[0]
            scaling_factor_1 = x_shape[1] / frost_shape[1]
            scaling_factor = np.maximum(scaling_factor_0, scaling_factor_1)

        scaling_factor *= 1.1
        new_shape = (int(np.ceil(frost_shape[1] * scaling_factor)),
                    int(np.ceil(frost_shape[0] * scaling_factor)))
        frost_rescaled = cv.resize(frost, dsize=new_shape,
                                    interpolation=cv.INTER_CUBIC)

        # randomly crop
        x_start, y_start = np.random.randint(0, frost_rescaled.shape[0] - x_shape[
            0]), np.random.randint(0, frost_rescaled.shape[1] - x_shape[1])

        if len(x_shape) < 3 or x_shape[2] < 3:
            frost_rescaled = frost_rescaled[x_start:x_start + x_shape[0],
                            y_start:y_start + x_shape[1]]
            frost_rescaled = rgb2gray(frost_rescaled)
        else:
            frost_rescaled = frost_rescaled[x_start:x_start + x_shape[0],
                            y_start:y_start + x_shape[1]][..., [2, 1, 0]]
            
        x1 = np.clip(c[0] * np.array(x1) + c[1] * frost_rescaled, 0, 255)
        x2 = np.clip(c[0] * np.array(x2) + c[1] * frost_rescaled, 0, 255)
        return x1, x2
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1, image2 = self.frost(image1, image2, self.severity)
            
        return image1, image2


class Snow:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'Snow@{severity}'
        
    def generate_snow_layer(self, image, c, angle):
        snow_layer = np.random.normal(size=image.shape[:2], loc=c[0],
                                    scale=c[1])  # [:2] for monochrome

        snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
        snow_layer[snow_layer < c[3]] = 0

        snow_layer = np.clip(snow_layer.squeeze(), 0, 1)
        snow_layer = _motion_blur(snow_layer, radius=c[4], sigma=c[5], angle=angle)

        # The snow layer is rounded and cropped to the img dims
        snow_layer = np.round(snow_layer * 255).astype(np.uint8) / 255.
        snow_layer = snow_layer[..., np.newaxis]
        snow_layer = snow_layer[:image.shape[0], :image.shape[1], :]
        
        return snow_layer
        
        
    def snow(self, x1, x2, severity):
        assert x1.shape == x2.shape
        
        x1 = Image.fromarray(x1)
        x2 = Image.fromarray(x2)
        
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
            (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
            (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
            (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
            (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

        x1 = np.array(x1, dtype=np.float32) / 255.
        x2 = np.array(x2, dtype=np.float32) / 255.
        
        snow_angle = np.random.uniform(-135, -45)
        snow_layer1 = self.generate_snow_layer(x1, c, snow_angle)
        snow_layer2 = self.generate_snow_layer(x2, c, snow_angle)
        
        # snow_layer = np.random.normal(size=x1.shape[:2], loc=c[0],
        #                             scale=c[1])  # [:2] for monochrome

        # snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
        # snow_layer[snow_layer < c[3]] = 0

        # snow_layer = np.clip(snow_layer.squeeze(), 0, 1)

        # snow_layer = _motion_blur(snow_layer, radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

        # # The snow layer is rounded and cropped to the img dims
        # snow_layer = np.round(snow_layer * 255).astype(np.uint8) / 255.
        # snow_layer = snow_layer[..., np.newaxis]
        # snow_layer = snow_layer[:x1.shape[0], :x1.shape[1], :]

        if len(x1.shape) < 3 or x1.shape[2] < 3:
            x1 = c[6] * x1 + (1 - c[6]) * np.maximum(x1, x1.reshape(x1.shape[0],
                                                                x1.shape[
                                                                    1]) * 1.5 + 0.5)
            x2 = c[6] * x2 + (1 - c[6]) * np.maximum(x2, x2.reshape(x2.shape[0],
                                                                x2.shape[
                                                                    1]) * 1.5 + 0.5)
            snow_layer1 = snow_layer1.squeeze(-1)
        else:
            x1 = c[6] * x1 + (1 - c[6]) * np.maximum(x1, cv.cvtColor(x1,
                                                                cv.COLOR_RGB2GRAY).reshape(
                x1.shape[0], x1.shape[1], 1) * 1.5 + 0.5)
            x2 = c[6] * x2 + (1 - c[6]) * np.maximum(x2, cv.cvtColor(x2,
                                                                cv.COLOR_RGB2GRAY).reshape(
                x2.shape[0], x2.shape[1], 1) * 1.5 + 0.5)
        try:
            x1 = np.clip(x1 + snow_layer1 + np.rot90(snow_layer1, k=2), 0, 1) * 255
            x2 = np.clip(x2 + snow_layer2 + np.rot90(snow_layer2, k=2), 0, 1) * 255
            return x1, x2
        except ValueError:
            print('ValueError for Snow, Exception handling')
            x1[:snow_layer1.shape[0], :snow_layer1.shape[1]] += snow_layer1 + np.rot90(
                snow_layer1, k=2)
            x2[:snow_layer2.shape[0], :snow_layer2.shape[1]] += snow_layer2 + np.rot90(
                snow_layer2, k=2)
            return np.clip(x1, 0, 1) * 255, np.clip(x2, 0, 1) * 255
        
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1, image2 = self.snow(image1, image2, self.severity)
            
        return image1, image2