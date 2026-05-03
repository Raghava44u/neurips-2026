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
- No-Connector: 0.680 EA (+21.0 pp) — confirms the connector paradox generalizes

---

### Requirements
- Python 3.10+
- CUDA GPU (≥16 GB VRAM recommended; 4-bit quantization supports 8 GB)
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


```

---
