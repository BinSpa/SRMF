# SRMF: A Data Augmentation and Multimodal Fusion Approach for Long-Tail UHR Satellite Image Segmentation

![Architecture Overview](images/main.jpg)

## Introduction

This repository contains the implementation of **SRMF**, a novel framework for semantic segmentation in ultra-high-resolution (UHR) satellite imagery. SRMF addresses the long-tail class distribution problem by incorporating a multi-scale cropping technique alongside a data augmentation strategy based on semantic reordering and resampling. The model also leverages multimodal fusion to integrate textual and visual features, which results in a more robust segmentation performance.

## Dataset Preparation

1. Download the datasets:
   - URUR: [Download link](https://github.com/jankyee/URUR.git)
   - GID: [Download link](https://x-ytong.github.io/project/GID.html)
   - FBP: [Download link](https://x-ytong.github.io/project/Five-Billion-Pixels.html)

2. Unzip the datasets and organize them as follows,  you need to manually convert the image parts of the GID and FBP datasets from four-channel images to RGB images:
```
urur/
  ├── train
  ├── image
  ├── label
  ├── val
  ├── image
  ├── label
  ├── test
  ├── image
  ├── label

gid/
  ├── train
    ├── rgb_images
    ├── gid_labels
  ├── val
    ├── rgb_images
    ├── gid_labels

fbp/
  ├── train
    ├── rgb_images
    ├── fbp_labels
  ├── val
    ├── rgb_images
    ├── fbp_labels
```

## Getting Started

### Prerequisites

- Python 3.10.13
- PyTorch 2.0.0
- timm 0.9.16
- mmcv 2.1.0
- mmengine 0.10.3

Or you can directly import the environment from the following YAML file:
``` conda env create -f srmf.yml ```

### Running the Code

1. Clone this repository:
   ```bash
   git clone https://github.com/username/srmf.git
   cd srmf
   ```

2. Train the model:
   ```bash
   python train.py --config configs/srmf_config.yaml
   ```

3. Test the model:
   ```bash
   python test.py --checkpoint path/to/checkpoint.pth
   ```

## Pre-trained Models

Download the pre-trained models from the following links:
- SRMF on URUR: [Download link](https://pan.baidu.com/s/15tmFcHH_4c-m5WnTIzgW6A?pwd=wuha), 
- SRMF on GID: [Download link](https://pan.baidu.com/s/1OyqkHJDtUFFW8JFBilOdNw?pwd=wbds)
- SRMF on FBP: [Download link](https://pan.baidu.com/s/1XgaNGx7d8NMsB_ceTGlyNQ?pwd=7xge)

Place the downloaded models in the `checkpoints/` directory:
```bash
mkdir checkpoints
mv path/to/downloaded_model checkpoints/
```

## Citation

If you find this work useful, please cite our paper:
```bibtex
@article{srmf2024,
  title={SRMF: A Data Augmentation and Multimodal Fusion Approach for Long-Tail UHR Satellite Image Segmentation},
  author={Yulong Guo, Zilun Zhang, Yongheng Shang, Yingchun Yang, Jianwei Yin, Tiancheng Zhao, Shuiguang Deng},
  journal={Journal of XYZ},
  year={2024}
}
```

## Acknowledgements

We thank the contributors and community for their invaluable support. This work is funded by the National Key Research and Development Program of China under grant number 2023YFD2000101 and the Hainan Province Science and Technology Special Fund (ZDYF2022SHFZ323).

