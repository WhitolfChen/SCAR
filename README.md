# SCAR

This is the official implementation of our paper [Taught Well Learned Ill: Towards Distillation-conditional Backdoor Attack](https://openreview.net/forum?id=IGowQfG5oA), accepted by NeurIPS 2025.

## Abstract

Knowledge distillation (KD) is a vital technique for deploying deep neural networks (DNNs) on resource-constrained devices by transferring knowledge from large teacher models to lightweight student models. While teacher models from third-party platforms may undergo security verification (e.g., backdoor detection), we uncover a novel and critical threat: distillation-conditional backdoor attacks (DCBAs). DCBA injects dormant and undetectable backdoors into teacher models, which become activated in student models via the KD process, even with clean distillation datasets. While the direct extension of existing methods is ineffective for DCBA, we implement this attack by formulating it as a bilevel optimization problem and proposing a simple yet effective method (i.e., SCAR). Specifically, the inner optimization simulates the KD process by optimizing a surrogate student model, while the outer optimization leverages outputs from this surrogate to optimize the teacher model for implanting the conditional backdoor. Our SCAR addresses this complex optimization utilizing an implicit differentiation algorithm with a pre-optimized trigger injection function. Extensive experiments across diverse datasets, model architectures, and KD techniques validate the effectiveness of our SCAR and its resistance against existing backdoor detection, highlighting a significant yet previously overlooked vulnerability in the KD process.

## Getting Started

### Installation

Install the required dependencies using `conda`:

```
conda env create -f environment.yaml
conda activate SCAR
```

### Usage

#### 1. Pre-optimize the Trigger Injection Function

Pre-optimize the trigger injection function on CIFAR-10 for ResNet-50 teacher model:
```
cd SCAR
python pretrain.py -d cifar10 -t resnet50 -g 0
```
The results can be found in folder `SCAR/pretrain/cifar10/resnet50`.


#### 2. Utilize SCAR for Attack

Train a ResNet-50 teacher model with a distillation-conditional backdoor on CIFAR-10:
```
python SCAR.py -d cifar10 -t resnet50 -g 0
```
The checkpoints of the compromised teacher model can be found in folder `SCAR/attack/cifar10/resnet50/ckp`.


#### 3. Evaluate the attack performance of SCAR

Evaluate the effectiveness of the SCAR attack on student models:
```
python test_distillation.py -d cifar10 -t resnet50 -s mobilenetv2 -m response -g 0
```
The results of student models can be found in folder `SCAR/distillation/cifar10/response/resnet50/mobilenetv2`.



## Citation

If you find our work useful for your research, please consider citing our paper:

```
@inproceedings{chen2025taught,
  title={Taught Well Learned Ill: Towards Distillation-conditional Backdoor Attack},
  author={Chen, Yukun and Li, Boheng and Yuan, Yu and Qi, Leyi and Li, Yiming and Zhang, Tianwei and Qin, Zhan and Ren, Kui},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```