from typing import Tuple
import numpy as np
import skimage as sk
from PIL import Image
from skimage.filters import gaussian
import subprocess
import json


class H264CRF:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'H264CRF@{severity}'
        
    def crf(self, video, out_path, severity, ffmpeg_root):
        c=[23,30,37,44,51][severity-1]
        return_code = subprocess.call(
        [f"{ffmpeg_root}/ffmpeg", "-i", video,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx264", "-crf", str(c), out_path])
        
        return return_code
    
    def __call__(self, src_video, out_video_path, ffmpeg_root):
        return_code = self.crf(src_video, out_video_path, self.severity, ffmpeg_root)
        return return_code
    

class H264ABR:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'H264ABR@{severity}'
        
    def abr(self, video, out_path, severity, ffmpeg_root):
        c=[2,4,8,16,32][severity-1]
        result = subprocess.Popen(
        [f"{ffmpeg_root}/ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

        data = json.load(result.stdout)

        bit_rate = str(int(50000000 / c))
        print(f'Original bit rate: {data["format"]["bit_rate"]}, New bit rate: {bit_rate}')

        return_code = subprocess.call(
            [f"{ffmpeg_root}/ffmpeg","-y", "-i", video,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx264", "-b:v", bit_rate, "-maxrate", bit_rate, "-bufsize",
            bit_rate, out_path])
        
        return return_code
    
    def __call__(self, src_video, out_video_path, ffmpeg_root):
        return_code = self.abr(src_video, out_video_path, self.severity, ffmpeg_root)
        return return_code
    

class BitError:
    def __init__(self, severity=5) -> None:
        self.severity = severity
        self.name = f'BitError@{severity}'
        
    def bit_error(self, video, out_path, severity, ffmpeg_root):
        # c=[100000, 50000, 30000, 20000, 10000][severity-1]
        c=[100000, 50000, 30000, 20000, 10000][severity-1] * 500
        return_code = subprocess.run(
            [f"{ffmpeg_root}/ffmpeg","-y", "-i", video, "-c", "copy", "-bsf:v", "noise={}".format(str(c)),
            out_path])
        
        return return_code
    
    def __call__(self, src_video, out_video_path, ffmpeg_root):
        return_code = self.bit_error(src_video, out_video_path, self.severity, ffmpeg_root)
        return return_code
