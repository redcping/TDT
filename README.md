
# Learning Triangular Distribution in Visual World (CVPR2024)

![license](https://img.shields.io/badge/License-MIT-brightgreen)
![python](https://img.shields.io/badge/Python-3.9-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.1-orange)

# Description
This repository is the official source of code for our paper titled "Learning Triangular Distribution in Visual World".  [Arxiv](https://arxiv.org/abs/2311.18605).

![image](https://github.com/redcping/TDT/assets/18466019/a617afbc-50a2-4395-bda6-0bba4c950563)



  We propose a so-called Triangular Distribution Transform(TDT) to build an injective function between feature and label, guaranteeing that any symmetric feature discrepancy linearly reflects the difference between labels. The proposed TDT can be used as a plug-in in mainstream backbone networks to address different label distribution learning tasks.


# Core Code
Please refer to the [tdt.py](https://github.com/redcping/TDT/blob/main/tdt.py) file. 
# Tasks Code
come soon
# Citation
If you find this work helpful in your research, please consider citing our paper:
```
@inproceedings{chen2024_TDT,
  title     = {Learning Triangular Distribution in Visual World},
  author    = {Ping Chen, Xingpeng Zhang, Chengtao Zhou, Dichao Fan, Peng Tu, Le Zhang and Yanlin Qian},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```
