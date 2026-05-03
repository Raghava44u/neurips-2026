# MemEIC — Memory-Augmented Multimodal Knowledge Editing

**NeurIPS 2026 Evaluations & Datasets Track**

[![arXiv](https://img.shields.io/badge/arXiv-2510.25798-b31b1b.svg)](https://arxiv.org/abs/2510.25798)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Datasets-yellow)](https://huggingface.co/datasets/MemEIC/CCKEB)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

---

## Overview

This repository contains the code, datasets, and evaluation scripts for:

- **CCKEB** (Compositional Continual Knowledge Editing Benchmark) — 6,278 image–text pairs for continual compositional multimodal knowledge editing (5,000 train / 1,278 eval).
- **Adversarial-2k** — 2,000-sample adversarial stress-test balanced across five cross-modal failure modes (400 per category).
- **MemEIC** — a memory-augmented multimodal knowledge editor with dual external memory and a brain-inspired gated cross-modal connector.

The paper introduces a failure-guided evaluation framework exposing systematic weaknesses in existing multimodal knowledge editing methods that standard benchmarks miss.

---

## Key Results

| Method | Edit Acc. | Rephrase Acc. | Portability | Locality |
|---|---|---|---|---|
| Human Baseline (n=3) | 0.925 | 0.940 | — | — |
| Pure RAG | 0.106 | 0.114 | 0.045 | 0.600 |
| Text-only RAG | 0.110 | 0.114 | 0.046 | 0.600 |
| MemEIC Baseline | 0.459 | 0.343 | 0.023 | 0.600 |
| MemEIC + AMG (Exp1) | 0.451 | 0.338 | 0.024 | 0.600 |
| MemEIC + STK (Exp2) | 0.461 | 0.345 | 0.025 | 0.600 |
| MemEIC + GC (Exp3) | 0.482 | 0.352 | 0.038 | 0.600 |
| **MemEIC + CBD (Exp4)** | **0.483** | **0.363** | 0.047 | 0.600 |
| MemEIC No-Connector | 0.482 | 0.353 | 0.046 | 0.600 |

**Per-category Edit Accuracy (Baseline vs Best):**

| Category | Baseline | Best (method) |
|---|---|---|
| F1 Polysemy | 0.793 | **0.794** (+ STK) |
| F2 Cross-Modal Conflict | 0.007 | **0.105** (No-connector, unsolved) |
| F3 Near-Miss Retrieval | 0.395 | **0.433** (No-connector) |
| F4 Multi-Hop Reasoning | 0.595 | **0.596** (+ STK) |
| F5 Hard Visual Distinction | 0.505 | **0.632** (+ CBD) |

**Cross-Architecture (Vicuna-7b-v1.5, 200 samples):**
- Baseline: 0.470 EA
- No-Connector: 0.680 EA (+21.0 pp) — confirms connector paradox generalises

---

## Project Structure

```
MemEIC/
│
├── README.md                         ← this file
├── requirements.txt                  ← core dependencies
├── requirements_pinned.txt           ← exact pinned versions
├── setup_environment.ps1             ← Windows environment setup script
│
├── datasets/                         ← All dataset files
│   ├── adversarial_2k.json           ← Primary benchmark (2,000 samples, 5 categories)
│   ├── adversarial_2k.json.bak       ← Backup before category field update
│   ├── CCKEB_eval.json               ← CCKEB evaluation split (1,278 pairs)
│   ├── CCKEB_train.json              ← CCKEB training split (5,000 pairs)
│   ├── complex_reasoning_dataset.json ← Adversarial V1 (intermediate)
│   ├── adversarial_reasoning_dataset.json ← Adversarial V2
│   ├── adversarial_v2_hard.json      ← Hard 200-sample subset used in Exp1-4
│   ├── adversarial_2k_stats.json     ← Dataset statistics
│   ├── _image_pool.json              ← MMKB entity image paths
│   ├── sample_preview.json           ← 10-sample preview
│   ├── adversarial_preview.json      ← Adversarial preview
│   ├── CCKEB_images/                 ← CCKEB entity images
│   ├── train2017/                    ← COCO Train2017 images (download separately)
│   └── prompt/                       ← Prompt templates
│
├── results/                          ← All experiment output files
│   ├── paper_checkpoint.json         ← Full 2000-sample baseline evaluation
│   ├── exp1_adaptive_gating.json     ← Exp1 AMG results
│   ├── exp2_soft_topk.json           ← Exp2 STK results
│   ├── exp3_consistency_connector.json ← Exp3 GC results (200-sample)
│   ├── exp4_confidence_threshold.json ← Exp4 CBD results
│   ├── cross_architecture_llava.json ← Vicuna-7b cross-arch validation
│   ├── paper_evidence_complete.json  ← Aggregated evidence for paper
│   ├── final_comparison.json         ← Original vs V1 vs V2 comparison
│   ├── ablation.json / ablation_study.json ← Ablation variants
│   ├── failure_cases.json            ← Failure case catalogue
│   ├── retrieval_sensitivity.json    ← Alpha sweep sensitivity analysis
│   ├── attention_heatmaps_layer30/31.png ← Attention visualisations
│   ├── plots/                        ← Generated result plots
│   └── ours_training/               ← Training logs and checkpoints
│
├── checkpoints/                      ← Model checkpoints (Git LFS)
│   ├── llava_stage1.pt               ← LLaVA Stage-1 weights
│   ├── llava_stage2/                 ← LLaVA Stage-2 LoRA adapters
│   ├── minigpt4_stage1.pt            ← MiniGPT-4 Stage-1 weights
│   ├── minigpt4_stage2/              ← MiniGPT-4 Stage-2 adapters
│   └── ours_model/                   ← MemEIC dual-memory model weights
│
├── hparams/                          ← Hyperparameter YAML configs per method
│   ├── FT/                           ← Fine-tuning LoRA configs
│   ├── IKE/                          ← In-Context Knowledge Editing
│   ├── LORA/                         ← LoRA adapter configs
│   ├── MEND/                         ← MEND meta-learning configs
│   ├── OURS/                         ← MemEIC configs
│   ├── SERAC/                        ← SERAC scope classifier configs
│   ├── TRAINING/                     ← Training hyperparameters
│   └── WISE/                         ← WISE dual-memory configs
│
├── easyeditor/                       ← EasyEditor library (modified)
│   ├── __init__.py
│   ├── dataset/                      ← Dataset loaders
│   ├── editors/                      ← Editor implementations (ROME, MEMIT, MEND, etc.)
│   ├── evaluate/                     ← Metric computation
│   ├── models/                       ← Model wrappers
│   ├── trainer/                      ← Training loop
│   └── util/                         ← Shared utilities
│
├── figs/                             ← Figures for MemEIC_NeurIPS2026.tex
│
├── dataset-part/                     ← CCKEB Dataset Paper (NeurIPS 2026 E&D Track)
│   ├── CCKEB_Dataset_Paper.tex       ← Main LaTeX paper
│   ├── checklist.tex                 ← NeurIPS submission checklist (fully filled)
│   ├── new-figs/                     ← All 8 paper figures (fig1–fig8)
│   └── generate_dataset_figures.py  ← Script to generate all 8 figures
│
├── NeurIPS/                          ← MemEIC system paper (NeurIPS 2026 main track)
│   └── paper/
│       └── MemEIC_NeurIPS2026.tex
│
│   ─── Dataset generation scripts ───────────────────────────────────────
├── generate_adversarial_2k.py        ← Generate Adversarial-2k (2,000 samples)
├── generate_adversarial_dataset.py   ← Generate Adversarial V1
├── generate_adversarial_v2_hard.py   ← Generate 200-sample hard subset
├── generate_train2017_2k.py          ← Generate 2k dataset from COCO Train2017
├── generate_plots.py                 ← Generate comparison/ablation plots
├── generate_visual_failures.py       ← Generate visual failure case images
│
│   ─── Experiment runner scripts ────────────────────────────────────────
├── run_neurips_experiments.py        ← Run ALL Exp1–5 on full Adversarial-2k
├── run_cross_architecture_validation.py ← Cross-arch validation (Vicuna-7b)
├── run_failure_analysis.py           ← Failure analysis pipeline (Exp3–7)
├── run_train2017_experiments.py      ← Run Exp1–4 on COCO Train2017 dataset
├── run_complete_study.py             ← End-to-end complete study runner
│
│   ─── Utility scripts ──────────────────────────────────────────────────
├── sample.py                         ← Quick sanity check / sampling
├── test_compositional_edit.py        ← Unit test for compositional edits
└── archive_candidates/               ← Archived debug scripts (not used in pipeline)
```

---

## Installation

### Requirements
- Python 3.10+
- CUDA GPU (≥16 GB VRAM recommended; 4-bit quantisation supports 8 GB)
- Windows (PowerShell) or Linux

### Step 1 — Create virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

For exact reproducibility (pinned versions):
```bash
pip install -r requirements_pinned.txt
```

Core packages:
- `torch`, `torchvision`, `torchaudio`
- `transformers`, `accelerate`, `peft`, `datasets`
- `sentence-transformers` (embedder: `all-MiniLM-L6-v2`)
- `open_clip_torch` (CLIP ViT-L/14 visual features)
- `scikit-learn`, `numpy`, `scipy`, `pandas`
- `matplotlib`, `tqdm`, `einops`, `pillow`

### Step 3 — COCO Train2017 (optional)

Only needed for the non-adversarial Train2017 experiments:
```bash
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d datasets/
```

---

## Datasets

### Adversarial-2k — `datasets/adversarial_2k.json`

| Field | Description |
|---|---|
| `category` | F1–F5 failure category |
| `failure_mode` | Human-readable mode name |
| `src` | Source question |
| `rephrase` | Rephrased question |
| `pred` | Correct prediction before edit |
| `alt` | Target answer after edit |
| `image` / `image_rephrase` | Visual input paths |
| `loc` / `loc_ans` | Locality probe Q&A |
| `m_loc_q` / `m_loc_a` / `m_loc_img` | Multimodal locality probe |
| `portability_q` / `portability_a` | Portability probe |
| `textual_edit` | Nested textual edit sub-object |

Five categories, 400 samples each (indices 0–399 = F1, 400–799 = F2, etc.):

| Code | Name | Description |
|---|---|---|
| F1 | Polysemy Attacks | Same word, different meaning in visual vs textual context |
| F2 | Cross-Modal Conflicts | Visual evidence contradicts edited text fact |
| F3 | Near-Miss Retrieval | Semantically close but wrong memory retrieved |
| F4 | Multi-Hop Reasoning | Chained inference over multiple knowledge steps |
| F5 | Hard Visual Distinctions | Visually similar but semantically distinct entities |

HuggingFace: `https://huggingface.co/datasets/MemEIC/CCKEB`

### CCKEB — `datasets/CCKEB_train.json` + `CCKEB_eval.json`

6,278 image–text pairs: 5,000 training + 1,278 evaluation.

---

## Reproducing Results

### Setup (Windows PowerShell)

```powershell
cd C:\Users\Dr-Prashantkumar\Downloads\MemEIC\MemEIC
.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
```

---

### Generate Adversarial-2k Dataset

> Skip if using the pre-built `datasets/adversarial_2k.json`.

```bash
python generate_adversarial_2k.py
# Output: datasets/adversarial_2k.json  (2,000 samples)
```

---

### Run All NeurIPS Experiments (Exp1–5)

```bash
python run_neurips_experiments.py
# Output: results/neurips_all_results.json
```

Runs in sequence: Baseline → AMG (Exp1) → STK (Exp2) → GC (Exp3) → CBD (Exp4) → CAFE (Exp5) → Pure RAG → Text-only RAG.

**Estimated runtime:** 8–12 hours on an A100 or RTX 3090.

---

### Pre-computed Results (already in `results/`)

All paper results are pre-computed. You can inspect them directly:

```python
import json

# Main table results (2000 samples)
with open("results/paper_checkpoint.json") as f:
    d = json.load(f)
# Fields per sample: cat, ce, cr, cp, be, br, bp, re, rr, rp, te, tr, tp, loc, dt
# ce/cr/cp = MemEIC Baseline (composition connector) edit/rephrase/portability  → EA=0.459
# be/br/bp = No-Connector ablation edit/rephrase/portability                    → EA=0.482
# re/rr/rp = Pure RAG edit/rephrase/portability                                 → EA=0.106
# te/tr/tp = Text-only RAG edit/rephrase/portability                            → EA=0.110
# loc       = Locality score (all methods = 0.600)

# Cross-architecture results
with open("results/cross_architecture_llava.json") as f:
    d = json.load(f)
print(d["methods"]["baseline"]["edit_acc"] / 100)   # 0.470
print(d["methods"]["no_connector"]["edit_acc"] / 100)  # 0.680

# Individual experiment results
for exp in ["exp1_adaptive_gating", "exp2_soft_topk",
            "exp3_consistency_connector", "exp4_confidence_threshold"]:
    with open(f"results/{exp}.json") as f:
        print(exp, json.load(f))
```

---

### Individual Experiment Descriptions

| Experiment | Script | Result File | Key Finding |
|---|---|---|---|
| Exp1: AMG | `run_neurips_experiments.py` | `exp1_adaptive_gating.json` | Adaptive gating −2 pp vs baseline |
| Exp2: STK | `run_neurips_experiments.py` | `exp2_soft_topk.json` | Soft top-k −4 pp overall, +12.5 pp on reasoning |
| Exp3: GC | `run_neurips_experiments.py` | `exp3_consistency_connector.json` | GC ≈ no-connector on Adversarial-2k; no-connector +13 pp on 200-sample hard subset |
| Exp4: CBD | `run_neurips_experiments.py` | `exp4_confidence_threshold.json` | τ=0.6 with 2% deferral: −1 pp |
| Cross-arch | `run_cross_architecture_validation.py` | `cross_architecture_llava.json` | Paradox holds on Vicuna-7b (+21 pp) |

---

### Cross-Architecture Validation

```bash
python run_cross_architecture_validation.py
# Output: results/cross_architecture_llava.json
# Requires: checkpoints/llava_stage2/, lmsys/vicuna-7b-v1.5 (auto-downloaded)
```

Uses 4-bit quantisation (`bitsandbytes`) to fit on a single GPU.

---

### Failure Analysis

```bash
python run_failure_analysis.py
# Outputs: results/failure_cases.json, results/final_comparison.json,
#          results/retrieval_sensitivity.json, results/ablation_study.json
```

---

### COCO Train2017 (Non-Adversarial) Experiments

```bash
python run_train2017_experiments.py
# Output: new-checkpoint/ (all 4 experiment result files)
```

Reports baseline 71.0% EA, no-connector 86.5% EA on clean COCO images.

---

### Generate Dataset Paper Figures

```bash
cd dataset-part
python generate_dataset_figures.py
# Output: new-figs/fig1_dataset_overview.png through fig8_qualitative_analysis.png
```

---

### Quick Sanity Check

```bash
python sample.py
# Prints 5 sample entries with correct fields

python test_compositional_edit.py
# Prints: compositional edit unit test PASSED
```

---

## Papers

### CCKEB Dataset Paper (NeurIPS 2026 E&D Track)

- **File:** `dataset-part/CCKEB_Dataset_Paper.tex`
- **Checklist:** `dataset-part/checklist.tex`
- **Figures:** `dataset-part/new-figs/`

Compile:
```bash
cd dataset-part
pdflatex CCKEB_Dataset_Paper.tex
bibtex CCKEB_Dataset_Paper
pdflatex CCKEB_Dataset_Paper.tex
pdflatex CCKEB_Dataset_Paper.tex
```

Overleaf: upload `CCKEB_Dataset_Paper.tex`, `checklist.tex`, `neurips_2026.sty`, and the `new-figs/` folder.

### MemEIC System Paper (NeurIPS 2026 Main Track)

- **File:** `NeurIPS/paper/MemEIC_NeurIPS2026.tex`
- **Figures:** `figs/`

Compile:
```bash
cd NeurIPS/paper
pdflatex MemEIC_NeurIPS2026.tex
bibtex MemEIC_NeurIPS2026
pdflatex MemEIC_NeurIPS2026.tex
pdflatex MemEIC_NeurIPS2026.tex
```

---

## Configuration

Hyperparameter configs in `hparams/` (YAML), one subfolder per method:

| Folder | Method |
|---|---|
| `FT/` | Fine-tuning with LoRA |
| `IKE/` | In-Context Knowledge Editing |
| `LORA/` | LoRA adapters |
| `MEND/` | MEND meta-learning editor |
| `OURS/` | MemEIC dual-memory (fusion α, gate threshold, memory size) |
| `SERAC/` | SERAC scope classifier |
| `TRAINING/` | Training (batch size, LR, warmup) |
| `WISE/` | WISE dual-memory (codebook size, update frequency) |

---

## Citation

```bibtex
@inproceedings{memeic2025,
  title     = {MemEIC: A Step Toward Continual and Compositional Knowledge Editing},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}

@inproceedings{cckeb2026,
  title     = {CCKEB and Adversarial-2k: Benchmarks for Compositional
               and Adversarial Multimodal Knowledge Editing},
  author    = {Anonymous},
  booktitle = {NeurIPS 2026 Evaluations and Datasets Track},
  year      = {2026}
}
```

---
