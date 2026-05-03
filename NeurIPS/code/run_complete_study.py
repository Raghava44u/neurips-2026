"""
MemEIC Complete Failure Analysis Study
=======================================
Steps 3-9: Run all experiments, collect metrics, generate plots,
           log failure cases, and produce publication-ready output.

Uses: SentenceTransformer (retrieval) + phi-2 (generation) on CUDA
"""

import json, random, os, time, torch, sys
from datetime import datetime
from statistics import mean
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# ── Config ──────────────────────────────────────────────────────────
ORIGINAL_PATH = "datasets/CCKEB_eval.json"
ADV_V1_PATH   = "datasets/adversarial_reasoning_dataset.json"
ADV_V2_PATH   = "datasets/adversarial_v2_hard.json"
N_ORIGINAL    = 50   # subsample from original
SEED          = 2026
random.seed(SEED)

# ── Load datasets ───────────────────────────────────────────────────
print("=" * 70)
print("  LOADING DATASETS")
print("=" * 70)

with open(ORIGINAL_PATH, "r", encoding="utf-8") as f:
    original_all = json.load(f)
with open(ADV_V1_PATH, "r", encoding="utf-8") as f:
    adv_v1_data = json.load(f)
with open(ADV_V2_PATH, "r", encoding="utf-8") as f:
    adv_v2_data = json.load(f)

original_data = random.sample(original_all, N_ORIGINAL)
print(f"  Original: {len(original_data)} samples (from {len(original_all)})")
print(f"  Adv V1:   {len(adv_v1_data)} samples")
print(f"  Adv V2:   {len(adv_v2_data)} samples")

# ── Load models ─────────────────────────────────────────────────────
print("\n  Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

MODEL_NAME = "microsoft/phi-2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Loading LLM ({MODEL_NAME}) on {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
    device_map=device,
)
if device != "cuda":
    model.to(device)
model.eval()
print("  Models loaded.\n")


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def build_memory(data):
    entries, embeddings = [], []
    for s in data:
        alt_val = s["alt"]
        if isinstance(alt_val, list):
            alt_val = alt_val[0]
        te = s["textual_edit"]
        te_alt = te["alt"][0] if isinstance(te["alt"], list) else te["alt"]
        te_pred = te["pred"][0] if isinstance(te["pred"], list) else te["pred"]
        entry = {
            "visual_q": s["src"], "visual_a": alt_val,
            "rephrase_q": s["rephrase"],
            "text_q": te["src"], "text_a": te_alt,
            "comp_q": s["port_new"][0]["Q&A"]["Question"],
            "comp_a": s["port_new"][0]["Q&A"]["Answer"],
            "loc_q": s["loc"], "loc_a": s["loc_ans"],
            "m_loc_q": s["m_loc_q"], "m_loc_a": s["m_loc_a"],
            "pred": s["pred"], "textual_pred": te_pred,
            "textual_rephrase": te["rephrase"],
            "textual_loc_q": te["loc"], "textual_loc_a": te["loc_ans"],
        }
        entries.append(entry)
        emb = embedder.encode(entry["visual_q"] + " " + entry["text_q"],
                              convert_to_tensor=True)
        embeddings.append(emb)
    return entries, embeddings


def retrieve(query, memory, mem_embs, alpha=0.9):
    q_emb = embedder.encode(query, convert_to_tensor=True)
    scores = [util.cos_sim(q_emb, emb).item() for emb in mem_embs]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return memory[best_idx], scores[best_idx], best_idx


def generate(prompt, max_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    return text.split("\n")[0].strip()


def check_answer(prediction, expected):
    p, e = prediction.lower().strip(), expected.lower().strip()
    return e in p or p in e


def build_edit_prompt(mem, question):
    return (f"Use the following facts to answer the question accurately.\n\n"
            f"Fact: {mem['visual_q']} → {mem['visual_a']}\n"
            f"Fact: {mem['text_q']} → {mem['text_a']}\n\n"
            f"Question: {question}\nAnswer:")


def build_simple_prompt(question):
    return f"Answer the following question accurately.\n\nQuestion: {question}\nAnswer:"


def build_no_retrieval_prompt(question):
    return f"Question: {question}\nAnswer:"


def build_no_connector_prompt(mem, question):
    return (f"Use this fact to answer.\n\n"
            f"Fact: {mem['visual_q']} → {mem['visual_a']}\n\n"
            f"Question: {question}\nAnswer:")


# ═══════════════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════

def evaluate_dataset(data, label, mode="full", alpha=0.9):
    print(f"\n{'─' * 60}")
    print(f"  {label} | mode={mode} | alpha={alpha} | n={len(data)}")
    print(f"{'─' * 60}")

    memory, mem_embs = build_memory(data)
    counters = {k: [] for k in [
        "edit_acc", "rephrase_acc", "locality_acc", "m_locality_acc",
        "portability_acc", "text_edit_acc", "text_rephrase_acc",
        "text_locality_acc", "retrieval_score", "baseline_acc",
    ]}
    details = []
    t0 = time.time()

    for i, mem in enumerate(memory):
        d = {"sample_idx": i, "src": mem["visual_q"], "expected": mem["visual_a"]}

        # Baseline
        bl = generate(build_simple_prompt(mem["visual_q"]))
        bl_ok = check_answer(bl, mem["visual_a"])
        counters["baseline_acc"].append(int(bl_ok))
        d["baseline"] = {"pred": bl, "correct": bl_ok}

        # Retrieve
        ret, score, ret_idx = retrieve(mem["visual_q"], memory, mem_embs, alpha)
        counters["retrieval_score"].append(score)
        d["retrieval_score"] = round(score, 4)
        d["self_retrieved"] = ret_idx == i

        # Build prompt based on mode
        if mode == "no_retrieval":
            prompt_fn = lambda q: build_no_retrieval_prompt(q)
        elif mode == "no_connector":
            prompt_fn = lambda q: build_no_connector_prompt(ret, q)
        else:
            prompt_fn = lambda q: build_edit_prompt(ret, q)

        # Edit
        e_out = generate(prompt_fn(mem["visual_q"]))
        e_ok = check_answer(e_out, mem["visual_a"])
        counters["edit_acc"].append(int(e_ok))
        d["edit"] = {"pred": e_out, "expected": mem["visual_a"], "correct": e_ok}

        # Rephrase
        r_out = generate(prompt_fn(mem["rephrase_q"]))
        r_ok = check_answer(r_out, mem["visual_a"])
        counters["rephrase_acc"].append(int(r_ok))
        d["rephrase"] = {"pred": r_out, "expected": mem["visual_a"], "correct": r_ok}

        # Locality
        l_out = generate(build_simple_prompt(mem["loc_q"]))
        l_ok = check_answer(l_out, mem["loc_a"])
        counters["locality_acc"].append(int(l_ok))
        d["locality"] = {"pred": l_out, "expected": mem["loc_a"], "correct": l_ok}

        # Multimodal locality
        ml_out = generate(build_simple_prompt(mem["m_loc_q"]))
        ml_ok = check_answer(ml_out, mem["m_loc_a"])
        counters["m_locality_acc"].append(int(ml_ok))
        d["m_locality"] = {"pred": ml_out, "expected": mem["m_loc_a"], "correct": ml_ok}

        # Portability
        p_out = generate(prompt_fn(mem["comp_q"]))
        p_ok = check_answer(p_out, mem["comp_a"])
        counters["portability_acc"].append(int(p_ok))
        d["portability"] = {"pred": p_out, "expected": mem["comp_a"], "correct": p_ok}

        # Text edit
        te_out = generate(prompt_fn(mem["text_q"]))
        te_ok = check_answer(te_out, mem["text_a"])
        counters["text_edit_acc"].append(int(te_ok))
        d["text_edit"] = {"pred": te_out, "expected": mem["text_a"], "correct": te_ok}

        # Text rephrase
        tr_out = generate(prompt_fn(mem["textual_rephrase"]))
        tr_ok = check_answer(tr_out, mem["text_a"])
        counters["text_rephrase_acc"].append(int(tr_ok))

        # Text locality
        tl_out = generate(build_simple_prompt(mem["textual_loc_q"]))
        tl_ok = check_answer(tl_out, mem["textual_loc_a"])
        counters["text_locality_acc"].append(int(tl_ok))

        details.append(d)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1:>3d}/{len(data)}] {elapsed:>6.0f}s | "
                  f"edit={mean(counters['edit_acc']):.2f} "
                  f"reph={mean(counters['rephrase_acc']):.2f} "
                  f"loc={mean(counters['locality_acc']):.2f} "
                  f"port={mean(counters['portability_acc']):.2f} "
                  f"ret={mean(counters['retrieval_score']):.3f}")

    elapsed = time.time() - t0
    summary = {k: round(mean(v), 4) for k, v in counters.items() if v}
    summary["eval_time_seconds"] = round(elapsed, 2)
    summary["num_samples"] = len(data)

    print(f"  Done in {elapsed:.0f}s")
    for k in ["edit_acc", "rephrase_acc", "locality_acc", "portability_acc", "retrieval_score"]:
        v = summary.get(k, 0)
        if k.endswith("_acc"):
            print(f"    {k:<25s}: {v*100:.1f}%")
        else:
            print(f"    {k:<25s}: {v:.4f}")

    return summary, details


# ═══════════════════════════════════════════════════════════════════
# ALPHA-BLENDED RETRIEVAL
# ═══════════════════════════════════════════════════════════════════

def build_memory_split(data):
    entries, vis_embs, txt_embs = [], [], []
    for s in data:
        alt_val = s["alt"] if isinstance(s["alt"], str) else s["alt"][0] if isinstance(s["alt"], list) else str(s["alt"])
        te = s["textual_edit"]
        te_alt = te["alt"][0] if isinstance(te["alt"], list) else te["alt"]
        te_pred = te["pred"][0] if isinstance(te["pred"], list) else te["pred"]
        entry = {
            "visual_q": s["src"], "visual_a": alt_val,
            "rephrase_q": s["rephrase"],
            "text_q": te["src"], "text_a": te_alt,
            "comp_q": s["port_new"][0]["Q&A"]["Question"],
            "comp_a": s["port_new"][0]["Q&A"]["Answer"],
            "loc_q": s["loc"], "loc_a": s["loc_ans"],
            "m_loc_q": s["m_loc_q"], "m_loc_a": s["m_loc_a"],
            "pred": s["pred"], "textual_pred": te_pred,
            "textual_rephrase": te["rephrase"],
            "textual_loc_q": te["loc"], "textual_loc_a": te["loc_ans"],
        }
        entries.append(entry)
        vis_embs.append(embedder.encode(entry["visual_q"], convert_to_tensor=True))
        txt_embs.append(embedder.encode(entry["text_q"], convert_to_tensor=True))
    return entries, vis_embs, txt_embs


def retrieve_alpha(query, memory, vis_embs, txt_embs, alpha):
    q_emb = embedder.encode(query, convert_to_tensor=True)
    scores = []
    for v_emb, t_emb in zip(vis_embs, txt_embs):
        vis_sim = util.cos_sim(q_emb, v_emb).item()
        txt_sim = util.cos_sim(q_emb, t_emb).item()
        blended = alpha * vis_sim + (1 - alpha) * txt_sim
        scores.append(blended)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return memory[best_idx], scores[best_idx], best_idx


# ═══════════════════════════════════════════════════════════════════
# STEP 3 & 4: RUN MAIN EXPERIMENTS + COLLECT METRICS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 3-4: MAIN EXPERIMENTS")
print("=" * 70)

os.makedirs("results/plots", exist_ok=True)

orig_sum, orig_det = evaluate_dataset(original_data, "Original CCKEB")
v1_sum, v1_det = evaluate_dataset(adv_v1_data, "Adv V1")
v2_sum, v2_det = evaluate_dataset(adv_v2_data, "Adv V2 (HARD)")

# Save individual results
for name, summary, details, path in [
    ("Original", orig_sum, orig_det, "results/original.json"),
    ("Adv V1", v1_sum, v1_det, "results/adv_v1.json"),
    ("Adv V2", v2_sum, v2_det, "results/adv_v2_hard.json"),
]:
    obj = {
        "metadata": {"timestamp": datetime.now().isoformat(),
                      "model": MODEL_NAME, "device": device, "label": name},
        "summary": summary,
        "sample_details": details,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# STEP 5: COMPARISON + PROVE FAILURE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 5: COMPARISON")
print("=" * 70)

metric_keys = [
    "edit_acc", "rephrase_acc", "locality_acc", "m_locality_acc",
    "portability_acc", "text_edit_acc", "retrieval_score",
]

comparison = {}
print(f"  {'Metric':<25s} {'Original':>10s} {'Adv V1':>10s} {'Adv V2':>10s}")
print(f"  {'-'*55}")
for k in metric_keys:
    o = orig_sum.get(k, 0)
    v1 = v1_sum.get(k, 0)
    v2 = v2_sum.get(k, 0)
    comparison[k] = {"original": o, "adv_v1": v1, "adv_v2_hard": v2}
    if k.endswith("_acc"):
        drop = (o - v2) * 100 if o > 0 else 0
        print(f"  {k:<25s} {o*100:>9.1f}% {v1*100:>9.1f}% {v2*100:>9.1f}%  (Δ={drop:+.1f}pp)")
    else:
        print(f"  {k:<25s} {o:>10.4f} {v1:>10.4f} {v2:>10.4f}")

with open("results/final_comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)
print("  Saved: results/final_comparison.json")


# ═══════════════════════════════════════════════════════════════════
# STEP 6: RETRIEVAL SENSITIVITY (α sweep on V2)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 6: RETRIEVAL SENSITIVITY")
print("=" * 70)

sensitivity_results = {}
alphas = [0.9, 0.5, 0.1]

mem_entries, vis_e, txt_e = build_memory_split(adv_v2_data)

for alpha in alphas:
    print(f"\n  Running Adv V2 with alpha={alpha}...")
    counters = {k: [] for k in [
        "edit_acc", "rephrase_acc", "locality_acc",
        "portability_acc", "retrieval_score",
    ]}
    t0 = time.time()

    for i, mem in enumerate(mem_entries):
        ret, score, ret_idx = retrieve_alpha(
            mem["visual_q"], mem_entries, vis_e, txt_e, alpha
        )
        counters["retrieval_score"].append(score)

        e_out = generate(build_edit_prompt(ret, mem["visual_q"]))
        counters["edit_acc"].append(int(check_answer(e_out, mem["visual_a"])))

        r_out = generate(build_edit_prompt(ret, mem["rephrase_q"]))
        counters["rephrase_acc"].append(int(check_answer(r_out, mem["visual_a"])))

        l_out = generate(build_simple_prompt(mem["loc_q"]))
        counters["locality_acc"].append(int(check_answer(l_out, mem["loc_a"])))

        p_out = generate(build_edit_prompt(ret, mem["comp_q"]))
        counters["portability_acc"].append(int(check_answer(p_out, mem["comp_a"])))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1:>3d}/{len(adv_v2_data)}] {elapsed:>6.0f}s | "
                  f"edit={mean(counters['edit_acc']):.2f} "
                  f"port={mean(counters['portability_acc']):.2f}")

    elapsed = time.time() - t0
    alpha_summary = {k: round(mean(v), 4) for k, v in counters.items() if v}
    alpha_summary["alpha"] = alpha
    alpha_summary["eval_time_seconds"] = round(elapsed, 2)
    sensitivity_results[f"alpha_{alpha}"] = alpha_summary

    print(f"    edit={alpha_summary['edit_acc']*100:.1f}% "
          f"port={alpha_summary['portability_acc']*100:.1f}% "
          f"loc={alpha_summary['locality_acc']*100:.1f}% "
          f"ret={alpha_summary['retrieval_score']:.4f} ({elapsed:.0f}s)")

with open("results/sensitivity.json", "w", encoding="utf-8") as f:
    json.dump(sensitivity_results, f, indent=2)
print("  Saved: results/sensitivity.json")


# ═══════════════════════════════════════════════════════════════════
# STEP 7: ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 7: ABLATION STUDY (on Adv V2)")
print("=" * 70)

ablation = {}

# Full model already computed
ablation["full_memeic"] = {k: v2_sum.get(k, 0) for k in metric_keys}

# Without retrieval
print("\n  [1/2] Without retrieval (direct question only)...")
abl1_sum, _ = evaluate_dataset(adv_v2_data, "V2-no-retrieval", mode="no_retrieval")
ablation["without_retrieval"] = {k: abl1_sum.get(k, 0) for k in metric_keys}

# Without connector (single fact only)
print("\n  [2/2] Without connector (single fact only)...")
abl2_sum, _ = evaluate_dataset(adv_v2_data, "V2-no-connector", mode="no_connector")
ablation["without_connector"] = {k: abl2_sum.get(k, 0) for k in metric_keys}

print(f"\n  ABLATION SUMMARY:")
print(f"  {'Metric':<25s} {'No Retriev':>10s} {'No Connect':>10s} {'Full':>10s}")
print(f"  {'-'*55}")
for k in metric_keys:
    vals = [ablation[m].get(k, 0) for m in ["without_retrieval", "without_connector", "full_memeic"]]
    if k.endswith("_acc"):
        print(f"  {k:<25s} {vals[0]*100:>9.1f}% {vals[1]*100:>9.1f}% {vals[2]*100:>9.1f}%")
    else:
        print(f"  {k:<25s} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}")

with open("results/ablation.json", "w", encoding="utf-8") as f:
    json.dump(ablation, f, indent=2)
print("  Saved: results/ablation.json")


# ═══════════════════════════════════════════════════════════════════
# STEP 8: FAILURE CASES (30+ from V2)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 8: FAILURE CASES")
print("=" * 70)

# The 200 V2 samples are shuffled. Determine category by textual edit loc question.
# Category markers based on loc questions set in generate script:
# ambiguity    → loc="What is the boiling point of water at sea level?"
# conflicting  → loc="What gas do plants absorb during photosynthesis?"
# retrieval    → loc="How many bones are in the adult human body?"
# multihop     → loc="What is the largest organ in the human body?"
# hard_dist    → loc="What planet do we live on?"

def get_failure_type(sample_data):
    loc = sample_data.get("loc", "")
    if "boiling point of water" in loc:
        return "ambiguity"
    elif "plants absorb" in loc:
        return "conflicting_signals"
    elif "bones" in loc:
        return "retrieval_error"
    elif "largest organ" in loc:
        return "reasoning_failure"
    elif "planet do we live" in loc:
        return "hard_distinction"
    return "memory_bias"

failure_cases = []
for d in v2_det:
    idx = d["sample_idx"]
    failed_metrics = []
    for metric in ["edit", "rephrase", "locality", "m_locality", "portability", "text_edit"]:
        if metric in d and not d[metric].get("correct", True):
            failed_metrics.append(metric)

    if failed_metrics:
        ftype = get_failure_type(adv_v2_data[idx])
        case = {
            "question": d["src"],
            "expected": d["expected"],
            "predicted": d.get("edit", {}).get("pred", "N/A"),
            "failure_type": ftype,
            "failed_metrics": failed_metrics,
            "num_failures": len(failed_metrics),
            "retrieval_score": d.get("retrieval_score", 0),
            "self_retrieved": d.get("self_retrieved", False),
            "sample_idx": idx,
        }
        failure_cases.append(case)

# Sort by severity (most failed metrics first)
failure_cases.sort(key=lambda x: x["num_failures"], reverse=True)

print(f"  Total failed samples: {len(failure_cases)}/{len(v2_det)}")
print(f"  Failure rate: {len(failure_cases)/len(v2_det)*100:.1f}%")
print(f"\n  Failure breakdown by type:")
type_counts = {}
for fc in failure_cases:
    t = fc["failure_type"]
    type_counts[t] = type_counts.get(t, 0) + 1
for t, c in sorted(type_counts.items()):
    print(f"    {t:<25s}: {c}")

print(f"\n  Top 10 most severe failures:")
for fc in failure_cases[:10]:
    print(f"    [{fc['sample_idx']:>3d}] {fc['failure_type']:<22s} "
          f"fails={fc['num_failures']}  ret={fc['retrieval_score']:.3f}")
    print(f"           Q: {fc['question'][:55]}")
    print(f"           exp={fc['expected'][:28]}  pred={fc['predicted'][:28]}")

with open("results/failure_cases.json", "w", encoding="utf-8") as f:
    json.dump(failure_cases, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: results/failure_cases.json ({len(failure_cases)} cases)")


# ═══════════════════════════════════════════════════════════════════
# STEP 9: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 9: GENERATING PUBLICATION PLOTS")
print("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication style
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]  # green, orange, red
LABELS = ["Original", "Adv V1", "Adv V2 (Hard)"]


# ---- PLOT 1: Edit Accuracy Bar Chart ----
fig, ax = plt.subplots(figsize=(8, 5))
metrics_to_plot = ["edit_acc", "rephrase_acc", "portability_acc", "text_edit_acc"]
metric_labels = ["Edit\nAccuracy", "Rephrase\nAccuracy", "Portability\nAccuracy", "Text Edit\nAccuracy"]
x = np.arange(len(metrics_to_plot))
width = 0.25

for j, (dset_key, color, label) in enumerate(zip(
    ["original", "adv_v1", "adv_v2_hard"], COLORS, LABELS
)):
    vals = [comparison[m][dset_key] * 100 for m in metrics_to_plot]
    bars = ax.bar(x + j * width, vals, width, label=label, color=color,
                  edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Accuracy (%)")
ax.set_title("MemEIC Edit Accuracy — Original vs Adversarial", fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 110)
ax.legend(loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/plots/edit_accuracy.png")
plt.close()
print("  Saved: results/plots/edit_accuracy.png")


# ---- PLOT 2: Locality Accuracy Comparison ----
fig, ax = plt.subplots(figsize=(7, 5))
loc_metrics = ["locality_acc", "m_locality_acc"]
loc_labels = ["Text Locality", "Multimodal Locality"]
x = np.arange(len(loc_metrics))
width = 0.25

for j, (dset_key, color, label) in enumerate(zip(
    ["original", "adv_v1", "adv_v2_hard"], COLORS, LABELS
)):
    vals = [comparison[m][dset_key] * 100 for m in loc_metrics]
    bars = ax.bar(x + j * width, vals, width, label=label, color=color,
                  edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("Accuracy (%)")
ax.set_title("Locality Preservation Under Adversarial Stress", fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(loc_labels)
ax.set_ylim(0, 110)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/plots/locality.png")
plt.close()
print("  Saved: results/plots/locality.png")


# ---- PLOT 3: Sensitivity Line Chart ----
fig, ax = plt.subplots(figsize=(8, 5))
alpha_vals = [0.1, 0.5, 0.9]
sens_metrics = ["edit_acc", "rephrase_acc", "locality_acc", "portability_acc", "retrieval_score"]
sens_labels = ["Edit Acc", "Rephrase Acc", "Locality Acc", "Portability Acc", "Retrieval Score"]
line_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
markers = ["o", "s", "D", "^", "v"]

for metric, mlabel, color, marker in zip(sens_metrics, sens_labels, line_colors, markers):
    vals = []
    for a in alpha_vals:
        key = f"alpha_{a}"
        v = sensitivity_results[key].get(metric, 0)
        vals.append(v * 100 if metric.endswith("_acc") else v)
    ax.plot(alpha_vals, vals, marker=marker, label=mlabel, color=color,
            linewidth=2.5, markersize=8)

ax.set_xlabel("α (Visual vs Textual Weight)")
ax.set_ylabel("Score (%)")
ax.set_title("Retrieval Sensitivity: Performance vs α", fontweight="bold")
ax.set_xticks([0.1, 0.5, 0.9])
ax.legend(loc="best")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/plots/sensitivity.png")
plt.close()
print("  Saved: results/plots/sensitivity.png")


# ---- PLOT 4: Ablation Study Bar Chart ----
fig, ax = plt.subplots(figsize=(8, 5))
abl_variants = ["without_retrieval", "without_connector", "full_memeic"]
abl_labels_x = ["No Retrieval", "No Connector", "Full MemEIC"]
abl_metrics = ["edit_acc", "rephrase_acc", "locality_acc", "portability_acc"]
abl_metric_labels = ["Edit", "Rephrase", "Locality", "Portability"]
abl_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

x = np.arange(len(abl_variants))
width = 0.2

for j, (metric, mlabel, color) in enumerate(zip(abl_metrics, abl_metric_labels, abl_colors)):
    vals = [ablation[v].get(metric, 0) * 100 for v in abl_variants]
    bars = ax.bar(x + j * width, vals, width, label=mlabel, color=color,
                  edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylabel("Accuracy (%)")
ax.set_title("Ablation Study: Component Contribution", fontweight="bold")
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(abl_labels_x)
ax.set_ylim(0, 110)
ax.legend(loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/plots/ablation.png")
plt.close()
print("  Saved: results/plots/ablation.png")


# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  ALL EXPERIMENTS COMPLETE")
print("=" * 70)
all_files = [
    "results/original.json",
    "results/adv_v1.json",
    "results/adv_v2_hard.json",
    "results/final_comparison.json",
    "results/sensitivity.json",
    "results/ablation.json",
    "results/failure_cases.json",
    "results/plots/edit_accuracy.png",
    "results/plots/locality.png",
    "results/plots/sensitivity.png",
    "results/plots/ablation.png",
]
print("  Output files:")
for f_path in all_files:
    exists = os.path.exists(f_path)
    status = "OK" if exists else "MISSING"
    print(f"    [{status}] {f_path}")

print("\n" + "=" * 70)
print("  KEY FINDINGS:")
print(f"    Edit accuracy drop:    {orig_sum.get('edit_acc',0)*100:.1f}% → {v2_sum.get('edit_acc',0)*100:.1f}%")
print(f"    Locality drop:         {orig_sum.get('locality_acc',0)*100:.1f}% → {v2_sum.get('locality_acc',0)*100:.1f}%")
print(f"    Portability drop:      {orig_sum.get('portability_acc',0)*100:.1f}% → {v2_sum.get('portability_acc',0)*100:.1f}%")
print(f"    Failure rate on V2:    {len(failure_cases)}/{len(v2_det)} ({len(failure_cases)/len(v2_det)*100:.1f}%)")
print(f"    Total failure cases:   {len(failure_cases)}")
print("=" * 70)
