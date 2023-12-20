# Point Transformer V3
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-transformer-v3-simpler-faster-stronger/lidar-semantic-segmentation-on-nuscenes)](https://paperswithcode.com/sota/lidar-semantic-segmentation-on-nuscenes?p=point-transformer-v3-simpler-faster-stronger)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-transformer-v3-simpler-faster-stronger/semantic-segmentation-on-s3dis)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis?p=point-transformer-v3-simpler-faster-stronger)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-transformer-v3-simpler-faster-stronger/semantic-segmentation-on-scannet)](https://paperswithcode.com/sota/semantic-segmentation-on-scannet?p=point-transformer-v3-simpler-faster-stronger)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-transformer-v3-simpler-faster-stronger/3d-semantic-segmentation-on-scannet200)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-scannet200?p=point-transformer-v3-simpler-faster-stronger)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-transformer-v3-simpler-faster-stronger/3d-semantic-segmentation-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=point-transformer-v3-simpler-faster-stronger)  

This repo is the official project repository of the paper **_Point Transformer V3: Simpler, Faster, Stronger_** and is mainly used for releasing schedules, updating instructions, sharing experiment records (contains model weight), and handling issues. The code will be updated in _[Pointcept](https://github.com/Pointcept/Pointcept) v1.5_.  
[ Backbone ] [PTv3] - [ [arXiv](https://arxiv.org/abs/2312.10035) ] [ [Bib](https://xywu.me/research/ptv3/bib.txt) ] [ [Code](https://github.com/Pointcept/Pointcept) ]  

<div align='left'>
<img src="assets/teaser.png" alt="teaser" width="800" />
</div>

## Highlights
- *Dec, 2023*: We released our project repo for PTv3, if you have any questions related to our work, please feel free to open an issue. Subscribe to our updates by filling out the [form](https://forms.gle/jHoBNqfhqK94WG678) and the subscription can be canceled by editing the form.

## Schedule
To make our polished code and reproduced experiments available as soon as possible, this time we will release what we already finished immediately after a validation instead of releasing them together after all work is done. We list a task list as follows:

- [ ] Release model code of PTv3;
- [ ] Release scratched config and record of indoor semantic segmentation;
  - [ ] ScanNet
  - [ ] ScanNet200
  - [ ] S3DIS
  - [ ] S3DIS 6-Fold (with cross-validation script) 
- [ ] Release pre-trained config and record of indoor semantic segmentation;
  - [ ] ScanNet (ScanNet + S3DIS + Structured3D)
  - [ ] ScanNet200 (Fine-tuned from above)
  - [ ] S3DIS (ScanNet + S3DIS + Structured3D)
  - [ ] S3DIS 6-Fold (Fine-tuned from ScanNet + Structured3D)
- [ ] Release scratched config and record of outdoor semantic segmentation;
  - [ ] NuScenes
  - [ ] SemanticKITTI
  - [ ] Waymo
- [ ] Release pre-trained config and record of outdoor semantic segmentation;
  - [ ] NuScenes (NuScenes + SemanticKITTI + Waymo)
  - [ ] SemanticKITTI (NuScenes + SemanticKITTI + Waymo)
  - [ ] Waymo (NuScenes + SemanticKITTI + Waymo)
- [ ] Release config and record of indoor instance segmentation;
  - [ ] ScanNet (Scratch and Fine-tuned from PPT pre-trained PTv3)
  - [ ] ScanNet200 (Scratch and Fine-tuned from PPT pre-trained PTv3)
- [ ] Release config and record of ScanNet data efficient benchmark;
- [ ] Release config and record of Waymo Object Detection benchmark;
- [ ] Release config and record of ImageNet classification;
  - [ ] ImageClassifier (making all 3D backbones in Pointcept support image classification)
  - [ ] Config and Record (PTv3 + SparseUNet)

## Citation
If you find _PTv3_ useful to your research, please cite our work as an acknowledgment. (੭ˊ꒳​ˋ)੭✧
```bib
@misc{wu2023ptv3,
      title={Point Transformer V3: Simpler, Faster, Stronger}, 
      author={Xiaoyang Wu and Li Jiang and Peng-Shuai Wang and Zhijian Liu and Xihui Liu and Yu Qiao and Wanli Ouyang and Tong He and Hengshuang Zhao},
      year={2023},
      eprint={2312.10035},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{wu2022ptv2,
    title={Point transformer V2: Grouped Vector Attention and Partition-based Pooling},
    author={Wu, Xiaoyang and Lao, Yixing and Jiang, Li and Liu, Xihui and Zhao, Hengshuang},
    booktitle={NeurIPS},
    year={2022}
}

@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished={\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```
