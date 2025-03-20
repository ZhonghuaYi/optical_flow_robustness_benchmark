import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def draw_psf(psf):
    # psf: 3, H, W
    psf = psf.cpu().numpy()
    psf = np.moveaxis(psf, 0, -1)
    # resize
    psf = cv.resize(psf, (512, 512), interpolation=cv.INTER_CUBIC)
    psf = cv.normalize(psf, None, 0, 255, cv.NORM_MINMAX)
    psf = psf.astype(np.uint8)
    psf = cv.cvtColor(psf, cv.COLOR_RGB2BGR)
    psf = psf[128:384, 128:384]
    return psf


psf_path_list = [
    'utils/corruption_util/psf/piece_6_fov20.0_fnum2.6_aper0_rms0.0296_idx8.pth',
    'utils/corruption_util/psf/piece_1_fov34.0_fnum3.8_aper0_rms0.0832_idx2.pth',
    'utils/corruption_util/psf/piece_6_fov28.0_fnum2.0_aper12_rms0.1102_idx8.pth',
    'utils/corruption_util/psf/piece_3_fov22.0_fnum2.0_aper4_rms0.1588_idx5.pth',
    'utils/corruption_util/psf/piece_4_fov20.0_fnum2.0_aper8_rms0.1939_idx4.pth'
]

num = 4

for index in range(len(psf_path_list)):
    psf = torch.load(psf_path_list[index])
    print(psf.shape)
    len_psf = len(psf)
    psfs = psf[::(len_psf // num)]
    for i in range(num):
        psf = psfs[i]
        psf = draw_psf(psf)
        name = f'outputs/PSF/{index}_{i}.png'
        cv.imwrite(name, psf)

