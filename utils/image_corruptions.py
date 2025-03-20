from .corruption_util.blur import *
from .corruption_util.digital import *
from .corruption_util.noise import *
from .corruption_util.weather import *
from .corruption_util.illumination import *
from .corruption_util.video import *


def get_corruption(corruption_name: str, severity: int, **kwargs):
    if corruption_name in ['JPEGCompression', 'jpeg_compression']:
        return JPEGCompression(severity=severity)
    elif corruption_name in ['Pixelation', 'pixelate']:
        return Pixelation(severity=severity)
    elif corruption_name in ['ElasticTransform', 'elastic_transform']:
        return ElasticTransform(severity=severity)
    elif corruption_name in ['Constrast', 'contrast']:
        return Constrast(severity=severity)
    elif corruption_name in ['Brightness', 'brightness']:
        return Brightness(severity=severity)
    elif corruption_name in ['LowLight', 'low_light']:
        return LowLight(severity=severity)
    elif corruption_name in ['Saturation', 'saturate']:
        return Saturation(severity=severity)
    elif corruption_name in ['OverExposure', 'over_exposure']:
        return OverExposure(severity=severity)
    elif corruption_name in ['UnderExposure', 'under_exposure']:
        return UnderExposure(severity=severity)
    elif corruption_name in ['LineDropout', 'line_dropout']:
        return LineDropout(severity=severity)
    elif corruption_name in ['Spatter', 'spatter']:
        return Spatter(severity=severity)
    elif corruption_name in ['Fog', 'fog']:
        return Fog(severity=severity)
    elif corruption_name in ['Frost', 'frost']:
        return Frost(severity=severity)
    elif corruption_name in ['Snow', 'snow']:
        return Snow(severity=severity)
    elif corruption_name in ['GaussianBlur', 'gaussian_blur']:
        return GaussianBlur(severity=severity)
    elif corruption_name in ['DefocusBlur', 'defocus_blur']:
        return DefocusBlur(severity=severity)
    elif corruption_name in ['GlassBlur', 'glass_blur']:
        return GlassBlur(severity=severity)
    elif corruption_name in ['CameraMotionBlur', 'camera_motion_blur']:
        return CameraMotionBlur(severity=severity)
    elif corruption_name in ['ObjectMotionBlur', 'object_motion_blur']:
        return ObjectMotionBlur(severity=severity)
    elif corruption_name in ['PSFBlur', 'psf_blur']:
        return PSFBlur(severity=severity, **kwargs)
    elif corruption_name in ['GaussianNoise', 'gaussian_noise']:
        return GaussianNoise(severity=severity)
    elif corruption_name in ['ShotNoise', 'shot_noise']:
        return ShotNoise(severity=severity)
    elif corruption_name in ['ImpulseNoise', 'impulse_noise']:
        return ImpulseNoise(severity=severity)
    elif corruption_name in ['SpeckleNoise', 'speckle_noise']:
        return SpeckleNoise(severity=severity)
    elif corruption_name in ['H264CRF', 'h264_crf']:
        return H264CRF(severity=severity)
    elif corruption_name in ['H264ABR', 'h264_abr']:
        return H264ABR(severity=severity)
    elif corruption_name in ['BitError', 'bit_error']:
        return BitError(severity=severity)
    else:
        raise ValueError(f'Unknown corruption_name: {corruption_name}')

