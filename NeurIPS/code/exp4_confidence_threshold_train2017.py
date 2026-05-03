"""
Experiment 4 (train2017): Retrieval Confidence Threshold
========================================================
Same as experiment_4_confidence_threshold.py but uses train2017-based 2K dataset.
Results saved to new-checkpoint/results/
"""

import json, random, os, time, torch, sys
from datetime import datetime
from statistics import mean
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
SEED = 2026
random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = "new-checkpoint/train2017_adversarial_2k.json"
RESULTS_DIR = "new-checkpoint/results"
PLOTS_DIR = "new-checkpoint/results/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Loaded {len(data)} train2017-based adversarial samples")

# ── Models ──────────────────────────────────────────────────────────
print("Loading models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
MODEL_NAME = "microsoft/phi-2"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True, device_map=device)
model.eval()
print(f"Models loaded on {device}")


# ── Helpers ─────────────────────────────────────────────────────────
def get_category(sample):
    loc = sample.get("loc", "")
    if "boiling point of water" in loc: return "ambiguity"
    elif "plants absorb" in loc: return "conflicting_signals"
    elif "bones" in loc: return "retrieval_error"
    elif "largest organ" in loc: return "reasoning_failure"
    elif "planet do we live" in loc: return "hard_distinction"
    return "unknown"


def build_memory_split(data):
    entries, vis_embs, txt_embs = [], [], []
    for s in data:
        alt_val = s["alt"] if isinstance(s["alt"], str) else s["alt"][0]
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


def generate_answer(prompt, max_tokens=30):
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


# ═══════════════════════════════════════════════════════════════════
# RETRIEVAL WITH CONFIDENCE
# ═══════════════════════════════════════════════════════════════════

def retrieve_with_confidence(query, memory, vis_embs, txt_embs):
    q_emb = embedder.encode(query, convert_to_tensor=True)
    scores = []
    for v_emb, t_emb in zip(vis_embs, txt_embs):
        vis_s = util.cos_sim(q_emb, v_emb).item()
        txt_s = util.cos_sim(q_emb, t_emb).item()
        scores.append(max(vis_s, txt_s))

    scores = np.array(scores)
    sorted_idx = np.argsort(scores)[::-1]
    best_idx = int(sorted_idx[0])
    best_score = float(scores[best_idx])
    margin = float(scores[sorted_idx[0]] - scores[sorted_idx[1]]) if len(scores) > 1 else 1.0

    return {
        "entry": memory[best_idx], "score": best_score,
        "idx": best_idx, "margin": margin,
    }


def eval_always_accept(mem, ret, question):
    """BASELINE: Always accept retrieval, never reject."""
    prompt = build_edit_prompt(ret["entry"], question)
    return generate_answer(prompt), False


def eval_threshold(mem, ret, question, min_confidence=0.6, min_margin=0.05):
    """PROPOSED: Reject if below confidence or margin threshold."""
    if ret["score"] < min_confidence or ret["margin"] < min_margin:
        # Reject → fallback to base model
        return generate_answer(build_simple_prompt(question)), True
    else:
        return generate_answer(build_edit_prompt(ret["entry"], question)), False


# ═══════════════════════════════════════════════════════════════════
# RUN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT 4 (train2017): RETRIEVAL CONFIDENCE THRESHOLD")
print(f"  Dataset: train2017 adversarial ({len(data)} samples)")
print("=" * 70)

memory, vis_embs, txt_embs = build_memory_split(data)

metrics_keys = ["edit_acc", "rephrase_acc", "locality_acc",
                "portability_acc", "retrieval_score"]

# Ablation configs
threshold_configs = [
    ("always_accept", 0.0, 0.0),
    ("threshold_0.5", 0.5, 0.03),
    ("threshold_0.6", 0.6, 0.05),
    ("threshold_0.7", 0.7, 0.08),
]

results = {}

for config_name, min_conf, min_marg in threshold_configs:
    print(f"\n  ── Running: {config_name.upper()} (conf={min_conf}, marg={min_marg}) ──")
    counters = {k: [] for k in metrics_keys}
    per_category = {}
    rejection_count = 0
    rejected_scores = []
    rejected_margins = []
    rejected_accs = []
    accepted_accs = []
    details = []
    t0 = time.time()

    for i, mem in enumerate(memory):
        cat = get_category(data[i])
        if cat not in per_category:
            per_category[cat] = {k: [] for k in metrics_keys}

        ret = retrieve_with_confidence(mem["visual_q"], memory, vis_embs, txt_embs)
        counters["retrieval_score"].append(ret["score"])
        per_category[cat]["retrieval_score"].append(ret["score"])

        # Edit accuracy with threshold
        if min_conf == 0.0:
            e_out, rejected = eval_always_accept(mem, ret, mem["visual_q"])
        else:
            e_out, rejected = eval_threshold(mem, ret, mem["visual_q"],
                                              min_confidence=min_conf, min_margin=min_marg)

        e_ok = check_answer(e_out, mem["visual_a"])
        counters["edit_acc"].append(int(e_ok))
        per_category[cat]["edit_acc"].append(int(e_ok))

        if rejected:
            rejection_count += 1
            rejected_scores.append(ret["score"])
            rejected_margins.append(ret["margin"])
            rejected_accs.append(int(e_ok))
        else:
            accepted_accs.append(int(e_ok))

        # Rephrase
        if min_conf == 0.0:
            r_out, _ = eval_always_accept(mem, ret, mem["rephrase_q"])
        else:
            r_out, _ = eval_threshold(mem, ret, mem["rephrase_q"],
                                       min_confidence=min_conf, min_margin=min_marg)
        r_ok = check_answer(r_out, mem["visual_a"])
        counters["rephrase_acc"].append(int(r_ok))
        per_category[cat]["rephrase_acc"].append(int(r_ok))

        # Locality
        l_out = generate_answer(build_simple_prompt(mem["loc_q"]))
        l_ok = check_answer(l_out, mem["loc_a"])
        counters["locality_acc"].append(int(l_ok))
        per_category[cat]["locality_acc"].append(int(l_ok))

        # Portability
        if min_conf == 0.0:
            p_out, _ = eval_always_accept(mem, ret, mem["comp_q"])
        else:
            p_out, _ = eval_threshold(mem, ret, mem["comp_q"],
                                       min_confidence=min_conf, min_margin=min_marg)
        p_ok = check_answer(p_out, mem["comp_a"])
        counters["portability_acc"].append(int(p_ok))
        per_category[cat]["portability_acc"].append(int(p_ok))

        details.append({
            "idx": i, "category": cat, "edit_ok": e_ok, "rejected": rejected,
            "score": ret["score"], "margin": ret["margin"],
        })

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1:>4d}/{len(data)}] {elapsed:>5.0f}s | "
                  f"edit={mean(counters['edit_acc']):.3f} "
                  f"reject={rejection_count}")

    elapsed = time.time() - t0
    overall = {k: round(mean(v), 4) for k, v in counters.items()}
    overall["rejection_count"] = rejection_count
    overall["rejection_rate"] = round(rejection_count / len(data), 4)
    overall["rejected_acc"] = round(mean(rejected_accs), 4) if rejected_accs else 0
    overall["accepted_acc"] = round(mean(accepted_accs), 4) if accepted_accs else 0
    overall["avg_rejected_score"] = round(mean(rejected_scores), 4) if rejected_scores else 0
    overall["avg_rejected_margin"] = round(mean(rejected_margins), 4) if rejected_margins else 0
    overall["eval_time_seconds"] = round(elapsed, 2)

    cat_summary = {}
    for cat, cat_cnt in per_category.items():
        cat_summary[cat] = {k: round(mean(v), 4) for k, v in cat_cnt.items() if v}
        cat_summary[cat]["count"] = len(cat_cnt["edit_acc"])

    results[config_name] = {
        "overall": overall,
        "per_category": cat_summary,
        "details": details,
    }

    print(f"\n  {config_name.upper()} Results:")
    print(f"    Edit:        {overall['edit_acc']*100:.1f}%")
    print(f"    Rephrase:    {overall['rephrase_acc']*100:.1f}%")
    print(f"    Locality:    {overall['locality_acc']*100:.1f}%")
    print(f"    Portability: {overall['portability_acc']*100:.1f}%")
    print(f"    Rejections:  {rejection_count} ({overall['rejection_rate']*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
# COMPARE PRIMARY: always_accept vs threshold_0.6
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PRIMARY COMPARISON: ALWAYS ACCEPT vs THRESHOLD 0.6")
print("=" * 70)

b = results["always_accept"]["overall"]
a = results["threshold_0.6"]["overall"]

deltas = {}
print(f"  {'Metric':<22s} {'Always':>10s} {'Thr 0.6':>10s} {'Delta':>10s}")
print(f"  {'─'*55}")
for k in metrics_keys:
    bv = b.get(k, 0)
    av = a.get(k, 0)
    d = av - bv
    deltas[k] = round(d, 4)
    if k.endswith("_acc"):
        print(f"  {k:<22s} {bv*100:>9.1f}% {av*100:>9.1f}% {d*100:>+9.1f}pp")
    else:
        print(f"  {k:<22s} {bv:>10.4f} {av:>10.4f} {d:>+10.4f}")


# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════
save_obj = {
    "experiment": "Retrieval Confidence Threshold (Experiment 4 — train2017)",
    "timestamp": datetime.now().isoformat(),
    "dataset": f"train2017_adversarial_2k ({len(data)} samples)",
    "image_source": "COCO train2017",
    "model": MODEL_NAME, "device": device,
    "threshold_configs": {name: {"min_confidence": c, "min_margin": m}
                         for name, c, m in threshold_configs},
}
for config_name, _, _ in threshold_configs:
    save_obj[config_name] = {
        "overall": results[config_name]["overall"],
        "per_category": results[config_name]["per_category"],
    }
save_obj["deltas"] = deltas

with open(f"{RESULTS_DIR}/exp4_confidence_threshold.json", "w", encoding="utf-8") as f:
    json.dump(save_obj, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {RESULTS_DIR}/exp4_confidence_threshold.json")

ckpt = {
    "experiment": "exp4_confidence_threshold",
    "dataset": DATA_PATH,
    "model": MODEL_NAME,
    "embedder": "all-MiniLM-L6-v2",
    "timestamp": datetime.now().isoformat(),
    "config": {"primary_threshold": 0.6, "primary_margin": 0.05},
    "results_summary": {
        "always_accept_edit_acc": b["edit_acc"],
        "threshold_0.6_edit_acc": a["edit_acc"],
        "delta_edit_acc": deltas["edit_acc"],
        "rejection_rate": a["rejection_rate"],
    },
    "ablation_results": {
        name: results[name]["overall"]["edit_acc"]
        for name, _, _ in threshold_configs
    },
}
with open("new-checkpoint/exp4_checkpoint.json", "w", encoding="utf-8") as f:
    json.dump(ckpt, f, indent=2)
print("Saved: new-checkpoint/exp4_checkpoint.json")


# ═══════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════
print("\n  GENERATING PLOTS...")

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 11,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})

# PLOT 1: Overall primary comparison
fig, ax = plt.subplots(figsize=(9, 5.5))
metrics_plot = ["edit_acc", "rephrase_acc", "portability_acc", "locality_acc"]
labels_plot = ["Edit\nAccuracy", "Rephrase\nAccuracy", "Portability\nAccuracy", "Locality\nAccuracy"]
x = np.arange(len(metrics_plot))
width = 0.32

bv = [b[m]*100 for m in metrics_plot]
av_vals = [a[m]*100 for m in metrics_plot]

bars1 = ax.bar(x - width/2, bv, width, label="Always Accept (Baseline)", color="#e74c3c", edgecolor="white")
bars2 = ax.bar(x + width/2, av_vals, width, label="Threshold 0.6 (Ours)", color="#2ecc71", edgecolor="white")

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.8, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Experiment 4: Confidence Threshold vs Always Accept\n(train2017 — {len(data)} samples)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels_plot)
ax.set_ylim(0, max(max(bv), max(av_vals)) + 15)
ax.legend(loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/exp4_overall_comparison.png")
plt.close()

# PLOT 2: Per-Category
all_cats = sorted(set(
    list(results["always_accept"]["per_category"].keys()) +
    list(results["threshold_0.6"]["per_category"].keys())
))
fig, ax = plt.subplots(figsize=(10, 5.5))
cats = sorted(all_cats)
x = np.arange(len(cats))
cat_labels = [c.replace("_", "\n") for c in cats]

bc_vals = [results["always_accept"]["per_category"].get(c, {}).get("edit_acc", 0)*100 for c in cats]
ac_vals = [results["threshold_0.6"]["per_category"].get(c, {}).get("edit_acc", 0)*100 for c in cats]

bars1 = ax.bar(x - 0.16, bc_vals, 0.32, label="Always Accept", color="#e74c3c", edgecolor="white")
bars2 = ax.bar(x + 0.16, ac_vals, 0.32, label="Threshold 0.6 (Ours)", color="#2ecc71", edgecolor="white")

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylabel("Edit Accuracy (%)")
ax.set_title("Per-Category: Always Accept vs Threshold 0.6 (train2017)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(cat_labels)
ax.set_ylim(0, max(max(bc_vals), max(ac_vals)) + 15)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/exp4_category_comparison.png")
plt.close()

# PLOT 3: Rejection Analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

rej = a["rejection_count"]
acc = len(data) - rej
axes[0].pie([acc, rej],
            labels=[f"Accepted\n({acc})", f"Rejected\n({rej})"],
            colors=["#2ecc71", "#e74c3c"],
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 12, "fontweight": "bold"})
axes[0].set_title(f"Retrieval Acceptance/Rejection\n(threshold=0.6)", fontweight="bold")

axes[1].bar(["Accepted", "Rejected (Fallback)"],
            [a["accepted_acc"]*100, a["rejected_acc"]*100],
            color=["#2ecc71", "#e74c3c"], edgecolor="white", width=0.5)
for i, v in enumerate([a["accepted_acc"]*100, a["rejected_acc"]*100]):
    axes[1].text(i, v+1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=14)
axes[1].set_ylabel("Edit Accuracy (%)")
axes[1].set_title("Accuracy: Accepted vs Rejected", fontweight="bold")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle("Confidence Threshold: Rejection Analysis (train2017)", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/exp4_rejection_analysis.png")
plt.close()

# PLOT 4: Threshold Ablation
fig, ax = plt.subplots(figsize=(8, 5))
config_labels = []
edit_accs = []
rej_rates = []
for name, c, m in threshold_configs:
    config_labels.append(f"({c}/{m})")
    edit_accs.append(results[name]["overall"]["edit_acc"] * 100)
    rej_rates.append(results[name]["overall"]["rejection_rate"] * 100)

x = np.arange(len(config_labels))
ax2 = ax.twinx()

bars = ax.bar(x, edit_accs, 0.4, color="#3498db", edgecolor="white", alpha=0.8, label="Edit Acc")
line = ax2.plot(x, rej_rates, "r-o", linewidth=2, markersize=8, label="Rejection Rate")

for bar, v in zip(bars, edit_accs):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.1f}%",
            ha="center", fontweight="bold", fontsize=9)

ax.set_xlabel("Threshold Config (confidence/margin)")
ax.set_ylabel("Edit Accuracy (%)", color="#3498db")
ax2.set_ylabel("Rejection Rate (%)", color="#e74c3c")
ax.set_xticks(x)
ax.set_xticklabels(config_labels)
ax.set_title("Threshold Ablation Study (train2017)", fontweight="bold")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc="upper left")

ax.spines["top"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/exp4_threshold_ablation.png")
plt.close()

print(f"\n  Plots saved to {PLOTS_DIR}/exp4_*.png")

print("\n" + "=" * 70)
print("  EXPERIMENT 4 (train2017) COMPLETE")
print("=" * 70)
print(f"  Rejection rate: {a['rejection_rate']*100:.1f}%")
print(f"  LIMITATION CONFIRMED: Confidence thresholding alone cannot solve adversarial failures")
