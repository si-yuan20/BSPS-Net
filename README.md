# BSPS‑Net

**Brain Hemispheric Symmetry Prior‑Guided Network for Hierarchical Difference Reasoning in Ischemic Stroke Lesion Segmentation**

> 
> Official repository for **BSPS‑Net**, a brain hemispheric symmetry prior‑guided framework for ischemic stroke lesion segmentation.

---

## Overview

Accurate segmentation of ischemic stroke lesions remains challenging because lesions may exhibit substantial variations in size, morphology, spatial distribution, and image contrast. Small or scattered lesions are particularly susceptible to missed detection, while indistinct boundaries can result in inaccurate delineation.

To address these challenges, we develop **BSPS‑Net**, a brain hemispheric symmetry prior‑guided segmentation network that explicitly exploits the approximate bilateral symmetry of the human brain. Instead of relying solely on features extracted from the original image, BSPS‑Net uses contralateral homologous brain tissue as an individualized anatomical reference and performs hierarchical cross‑hemispheric difference reasoning for lesion identification and delineation.

The framework is built upon **nnU‑Net v2** and introduces dedicated modules for homologous alignment, hierarchical differential representation, global relationship reasoning, and symmetry‑aware supervision.

![](docs/images/network_architecture.png)
  

*Figure 1: Overall architecture of the proposed BSPS‑Net framework*

![](docs/images/dataset_sample.png)
  

*Figure 2: Example MRI samples and corresponding lesion annotations from the dataset*


---

## Method

The overall BSPS‑Net framework follows the reasoning paradigm:
**Homologous Correspondence → Hierarchical Difference Reasoning → Symmetry‑aware Supervision**

The main components include:

### CG‑BLPA

**Correlation‑Guided Bidirectional Lesion‑Preserving Homologous Alignment**
CG‑BLPHA establishes homologous correspondence between the bilateral cerebral hemispheres through local correlation‑guided bidirectional matching while reducing the influence of lesion‑related abnormal signals during alignment.

### H‑DRE

**Hierarchical Differential Representation and Global Relation Reasoning**
H‑DRE models pathological differences between homologous hemispheric regions at multiple representation levels, including:

- low‑level signal and boundary differences;
- mid‑level structural and textural differences;
- high‑level semantic and contextual differences;
- global relationships between homologous cerebral regions.

### HRSC‑Loss

**Hemispheric Relation‑Symmetry Constraint Loss**
HRSC‑Loss explicitly constrains the relationship between bilateral homologous brain regions during network optimization. It encourages consistency in normal homologous tissue while maintaining discriminative pathological differences in lesion regions.

---

## Framework

BSPS‑Net is implemented based on **nnU‑Net v2**.
The nnU‑Net v2 pipeline is used as the fundamental medical image segmentation framework for:

- dataset preprocessing;
- experiment planning;
- data augmentation;
- patch‑based training;
- deep supervision;
- inference;
- post‑processing;
- evaluation.

The proposed hemispheric symmetry modeling components are integrated into the nnU‑Net v2 training and inference framework.

---

## Datasets

BSPS‑Net is evaluated on public ischemic stroke lesion segmentation datasets, including:

### ISLES 2022

The **ISLES 2022** dataset is used for evaluating ischemic stroke lesion segmentation on diffusion‑weighted MRI.

### ATLAS v2.0

The **ATLAS v2.0** dataset is used as an additional benchmark to evaluate lesion segmentation performance and generalization under different imaging characteristics.

The preprocessing configurations and experiment plans for different datasets are independently determined according to their respective image characteristics.

> 
> **Note:** This repository does not redistribute the original datasets.
> Users should obtain the datasets from their official sources and comply with the corresponding licenses and data‑use agreements.

---

## Repository Status

🚧 **This repository is currently a placeholder for the official implementation of BSPS‑Net.**

To ensure consistency between the released implementation and the final published manuscript, the complete reproducibility package will be made publicly available **after the manuscript is accepted for publication**.

Upon paper acceptance, we will release:

- Complete BSPS‑Net source code
- CG‑BLPHA implementation
- H‑DRE implementation
- HRSC‑Loss implementation
- nnU‑Net v2 integration
- Training scripts
- Inference scripts
- Evaluation scripts
- Data preprocessing scripts
- Preprocessing parameters and nnU‑Net experiment plan files
- Dataset split information
- Training/validation/test split files
- Complete training configurations
- Hyperparameter settings
- nnU‑Net v2 plans and configuration files
- Model checkpoints / trained weights
- Reproduction instructions
- Quantitative evaluation scripts
- Visualization scripts
- Environment and dependency specifications

> 
> ⚠️ **Important Statement**: Model weights, dataset‑specific preprocessing parameter files, experiment plan files and train/val/test partition files are not available at present. All above resources will be open‑sourced synchronously after the paper is formally accepted. We will also update the README and release corresponding assets in repository release page at that time.

---

## Model Weights

Pre‑trained model weights are **not publicly available at this stage**.
After acceptance of the manuscript, trained model checkpoints corresponding to the experiments reported in the paper will be released together with the source code whenever permitted by the applicable data‑use and repository policies.

The release will include sufficient configuration information to reproduce the reported experiments.

---

## Dataset Splits

The exact patient‑level dataset partitions used in the manuscript are not released in the placeholder version of this repository.
After manuscript acceptance, we plan to provide the corresponding dataset split information, including:

```
train
validation
test
```

where permitted by the corresponding dataset policies.
Only case identifiers or split definition files will be provided. **Original medical images and annotations will not be redistributed through this repository.**

---

## Training Configuration

Detailed experimental configurations will be released together with the final implementation, including:

```
Framework
Network configuration
Input configuration
Target spacing
Patch size
Batch size
Optimizer
Initial learning rate
Learning‑rate scheduler
Number of epochs
Data augmentation
Deep supervision
Loss configuration
Random seed
Mixed‑precision configuration
Inference configuration
Post‑processing configuration
Hardware environment
Software environment
```

The corresponding **nnU‑Net v2 plans**, training configurations, preprocessing parameters and necessary experiment metadata will also be provided to facilitate reproducibility.

## Reproducibility

Reproducibility is an important objective of this project.
Following manuscript acceptance, we intend to provide the necessary materials for reproducing the principal experiments reported in the paper, including:
**Code + Configuration + Preprocessing Parameters + Dataset Splits + Model Weights + Evaluation Scripts**

Because BSPS‑Net is developed on top of nnU‑Net v2, the released implementation will retain compatibility with the corresponding nnU‑Net v2 data organization and experiment workflow whenever possible.

---

## Citation

If you find this work useful, please consider citing our paper.
The formal citation and BibTeX entry will be updated after publication.

```
@article{BSPSNet2026,
  title   = {Brain Hemispheric Symmetry Prior‑Guided Network for Hierarchical Difference Reasoning in Ischemic Stroke Lesion Segmentation},
  author  = {To be updated},
  journal = {To be updated},
  year    = {2026}
}
```

> 
> The citation information above is currently a placeholder and will be replaced by the official bibliographic information after publication.

---

## Acknowledgements

This work is developed based on the **nnU‑Net v2** medical image segmentation framework.
We gratefully acknowledge the developers and contributors of nnU‑Net as well as the organizers and contributors of the public datasets used in this study.

---

## License

The license for the BSPS‑Net implementation will be specified when the complete source code is officially released.
Please note that external datasets, nnU‑Net v2, and other third‑party components remain subject to their respective licenses and terms of use.

---

## Updates

Major updates regarding code, model weights, preprocessing parameters, dataset splits, and reproducibility materials will be announced through this repository.
⭐ **You may star or watch this repository to follow future releases.**

---

## Contact

For questions regarding this work, please use the **Issues** section of this GitHub repository.

---

**BSPS‑Net © 2026**

