---
license: apache-2.0
task_categories:
- visual-question-answering
language:
- en
tags:
- multimodal
- knowledge-editing
pretty_name: CCKEB
configs:
- config_name: default
  data_files:
  - split: train
    path: CCKEB_train.json
  - split: test
    path: CCKEB_eval.json
---

# CCKEB (Compositional/Continual Knowledge Editing Benchmark)

[![arXiv](https://img.shields.io/badge/arXiv-2510.25798-b31b1b.svg)](https://arxiv.org/abs/2510.25798)
[![GitHub](https://img.shields.io/badge/GitHub-MemEIC-blue.svg)](https://github.com/MemEIC/MemEIC)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/MemEIC/CCKEB)

## 🌟 Overview
**CCKEB** is a benchmark designed for **Continual and Compositional Knowledge Editing** in Large Vision-Language Models (LVLMs), accepted at **NeurIPS 2025**.  

The benchmark targets realistic knowledge update scenarios in which **visual identities** and **textual facts** are edited **sequentially**.  
Models are required to retain previously edited knowledge while answering **compositional multimodal queries** that depend on both updated visual and textual information.

CCKEB evaluates two core capabilities:
- **Knowledge retention** under continual edits
- **Compositional reasoning**, i.e., integrating edited visual and textual knowledge to answer complex queries

To assess this, CCKEB introduces **Compositional Reliability (CompRel)**,  
which measures whether a model can correctly answer queries that require combining multiple edited knowledge pieces across modalities.


## 📊 Dataset Statistics

- **Total instances**: 6,278 visual–textual editing pairs  
- **Training set**: 5,000 pairs  
- **Evaluation set**: 1,278 pairs  

Each instance is constructed as a paired visual–textual edit targeting the same entity, 
and consists of:
- an image,
- a visual identity edit,
- a textual factual edit, and
- visual, textual, and compositional QA pairs.


## 🚀 Quick Start
You can easily load this dataset with the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the CCKEB dataset
dataset = load_dataset("MemEIC/CCKEB")

# Access train/test splits
print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")
print(dataset['train'][0])
```

## 📜 License
This dataset is released under the **Apache License 2.0**.

It is partially derived from the **VLKEB** dataset, which is licensed under the BSD 3-Clause License. All original copyright notices are preserved.

## 🖊️ Citation
If you use this dataset, please cite our paper:

```bibtex
@inproceedings{
seong2025memeic,
title={Mem{EIC}: A Step Toward Continual and Compositional Knowledge Editing},
author={Jin Seong and Jiyun Park and Wencke Liermann and Hongseok Choi and Yoonji Nam and Hyun Kim and Soojong Lim and Namhoon Lee},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=Qvj8s2rRUs}
}
```

## Aknowledgement
This work was supported by the Institute of Information & Communications Technology Planning
& Evaluation (IITP) grant funded by the Korea Government (MSIT) (No. RS-2023-00216011,
Development of Artificial Complex Intelligence for Conceptually Understanding and Inferring like
Human). It was also supported by the IITP grant funded by the Korea Government (MSIT) (No.
RS-2024-00338140, Development of learning and utilization technology to reflect sustainability of
generative language models and up-to-dateness over time)

### Related Works
We also encourage citing the foundational works this benchmark builds upon:

- **VLKEB**: [(NeurIPS'24) VLKEB: A Large Vision-Language Model Knowledge Editing Benchmark](https://github.com/VLKEB/VLKEB)
- **EasyEdit**: [An easy-to-use knowledge editing framework for large language models](https://github.com/zjunlp/EasyEdit)
