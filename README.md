# PepR Score 
![License](https://img.shields.io/github/license/PedramBakh/ec-nas-bench)
[![arXiv](results/arxiv.svg)](https://arxiv.org/abs/2403.09441)

## Abstract
The recent advances in deep learning (DL) have been accelerated by access to large-scale data and compute. These large-scale resources have been used to train progressively larger models which are resource intensive in terms of compute, data, energy, and carbon emissions. These costs are becoming a new type of entry barrier to researchers and practitioners with limited access to resources at such scale, particularly in the _Global South_. In this work, we take a comprehensive look at the landscape of existing DL models for vision tasks and demonstrate their usefulness in settings where resources are limited. To account for the resource consumption of DL models, we introduce a novel measure to estimate the performance per resource unit, which we call the PePR score.  Using a diverse family of 131 unique DL architectures (spanning $1M$ to $130M$ trainable parameters) and three medical image datasets, we capture trends about the performance-resource trade-offs. In applications like medical image analysis, we argue that small-scale, specialized  models are better than striving for large-scale models. Furthermore, we show that using pretrained models can significantly reduce the computational resources and data required. We hope this work will encourage the community to focus on improving AI equity by developing methods and models with smaller resource footprints.

<p float="left">
  <img src="results/github.png" width="800" height="" />
</p>

## Citation
Kindly use the following BibTeX entry if you use the code in your work.
```
@inproceedings{selvan2025pepr,
  title={PePR: Performance Per Resource Unit as a Metric to Promote Small-scale Deep Learning},
  author={Selvan, Raghavendra and Pepin, Bob and Igel, Christian and Samuel, Gabrielle and Dam, Erik B},
  booktitle={Northern Lights Deep Learning Conference},
  pages={220--229},
  year={2025},
  organization={PMLR}
}
```
## Requirements

* Standard Pytorch requirements to train the models. 
* TIMM library for using the specific architectures.

## Example Usage

Recreate paper plots.

```
python paper_plot.py
```

