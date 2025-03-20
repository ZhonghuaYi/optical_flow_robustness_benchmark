# optical_flow_robustness_benchmark
Official repository for "Benchmarking the Robustness of Optical Flow Estimation to Corruptions"


## TODO
- [ ] Provide evaluate code for all models mentioned in the paper.
- [ ] Provide easy-to-use interface to evaluate optical flow models.
- [ ] Provide low disk usage version of KITTI-FC and GoPro-FC.


## Setup

### Dataset preparation

KITTI-FC can be downloaded from [OpenDataLab](https://openxlab.org.cn/datasets/kaedeaki/KITTI-FC/tree/main) (43.92GB).

GoPro-FC can be downloaded from [OpenDataLab](https://openxlab.org.cn/datasets/kaedeaki/GoPro-FC/tree/main), with 30FPS version (680.93GB) and 10FPS version (529.38GB).

Extract them into the `data` folder.

**Note1**: the corrupted data contains 'line_dropout', 'elastic_transform' and 'speckle_noise' three data folder, but they are not used in our benchmark.

**Note2**: the 'high_light' corrupted data is in 'brightness' folder, as it comes from the [image corruptions library](https://github.com/bethgelab/imagecorruptions).

### Enviroment setup

Use conda to create a enviroment from `env.yaml`:
```bash
conda env create -f env.yaml
```

## Usage

### Benchmark the optical flow model

Sample code to evaluate RAFT model in KITTI-FC and GoPro-FC are provided in `models/RAFT/benchmark.py` and `models/RAFT/benchmark_gopro.py`.
To evaluate more models, refer these code.

### Corrupted data generation

We provide sample code which are used to create KITTI-FC and GoPro-FC in `KITTI_corrupt.py` and `GoPro_corrupt.py`.


## Cite
```tex
@article{yi2024benchmarking,
  title={Benchmarking the Robustness of Optical Flow Estimation to Corruptions},
  author={Yi, Zhonghua and Shi, Hao and Jiang, Qi and Gao, Yao and Wang, Ze and Zhang, Yufan and Yang, Kailun and Wang, Kaiwei},
  journal={arXiv preprint arXiv:2411.14865},
  year={2024}
}
```

