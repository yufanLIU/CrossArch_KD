# Cross-Architecture Knowledge Distillation

This is an open source implementation of the paper called "[Cross-Architecture Knowledge Distillation](https://openaccess.thecvf.com/content/ACCV2022/html/Liu_Cross-Architecture_Knowledge_Distillation_ACCV_2022_paper.html)".

> Yufan Liu, Jiajiong Cao, Bing Li, Weiming Hu, Jingting Ding, Liang Li. Cross-Architecture Knowledge Distillation. Asian Conference on Computer Vision (ACCV): Long Beach, China. 2022.12.04-2022.12.08.

## Abstract
*Transformer attracts much attention because of its ability to learn global relations and superior performance. In order to achieve higher performance, it is natural to distill complementary knowledge from Transformer to convolutional neural network (CNN). However, most existing knowledge distillation methods only consider homologous-architecture distillation, such as distilling knowledge from CNN to CNN. They may not be suitable when applying to cross-architecture scenarios, such as from Transformer to CNN. To deal with this problem, a novel cross-architecture knowledge distillation method is proposed. Specifically, instead of directly mimicking output/intermediate features of the teacher, partially cross attention projector and group-wise linear projector are introduced to align the student features with the teacher's in two projected feature spaces. And a multi-view robust training scheme is further presented to improve the robustness and stability of the framework. Extensive experiments
show that the proposed method outperforms 14 state-of-the-arts on both smallscale and large-scale datasets.*

## Citation
If you find our method useful in your research, please cite our paper: 
````
@InProceedings{Liu_2022_ACCV,
    author    = {Liu, Yufan and Cao, Jiajiong and Li, Bing and Hu, Weiming and Ding, Jingting and Li, Liang},
    title     = {Cross-Architecture Knowledge Distillation},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2022},
    pages     = {3396-3411}
}
````

## Framework
![Framework](./img/framework.png "Framework")

## Requirements

- The codes are tested on torch==1.12.0 on cuda==11.3 with 8 V100 (A100) cards.
- torch >= 1.12.0
- cuda >= 11.3

## Usage
- step 1: obtain your TORCHVISION_MODEL_PATH and TORCH_NN_PATH
```python
import os
import torchvision
os.systm('export TORCHVISION_MODEL_PATH={:s}'.format(torchvision.models.__path__[0]))
import torch
os.systm('export TORCH_NN_PATH={:s}'.format(torch.nn.__path__[0]))
```
- step2: make modifications to the native torch to extract necessary mid-level outputs for CAKD
```shell
cd CAKD
cp cakd_modified_files/resnet.py ${TORCHVISION_MODEL_PATH}/resnet.py
cp cakd_modified_files/vision_transformer.py ${TORCHVISION_MODEL_PATH}/vision_transformer.py
cp cakd_modified_files/functional.py ${TORCH_NN_PATH}/functional.py
```

- step3: run experiments
```shell
#run student baseline
sh experiments/run_baseline.sh
```
```shell
#run logits KD
sh experiments/run_logits.sh
```
```shell
#run logits CAKD
sh experiments/run_cakd.sh
```

## Notice and Performance
The performance reported in the original paper is produced based on a customized torch (with many customized data augmentation techniques). At this end, the performance of the student, teacher, and distilled student is relatively higher than the public models. Unfortunately, because of privacy and security concerns, we are not able to provide the full version of this torch. Instead, we make the key codes of CAKD public in this repo.

Since some customized operations are not available, the performance of the student, teacher, and distilled student is lower than that reported in the paper. However, the performance gain compared with competing methods is significant. The performance is provided below, researchers may consider the reproduced performance in this repo for fair comparisons.


**Performance on ImageNet.**
|        Method        |   Top-1  |   Top-5  |
|        :----:        |  :----:  |  :----:  |
| Baseline (ResNet50)  |  73.82%  |  91.97%  |
|       Logits         |  74.48%  |  92.29%  |
|    CAKD (Ours)       |**76.21%**|**93.09%**|




## Contact
If any question, please contact yufan.liu@ia.ac.cn or use public issues section of this repository.
