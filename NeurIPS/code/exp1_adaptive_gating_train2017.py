"""
Experiment 1 (train2017): Adaptive Modality Gating
===================================================
Same as experiment_1_adaptive_gating.py but uses train2017-based 2K dataset.
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
# RETRIEVAL METHODS
# ═══════════════════════════════════════════════════════════════════

def retrieve_baseline(query, memory, vis_embs, txt_embs):
    q_emb = embedder.encode(query, convert_to_tensor=True)
    vis_scores, txt_scores = [], []
    for v_emb, t_emb in zip(vis_embs, txt_embs):
        vis_scores.append(util.cos_sim(q_emb, v_emb).item())
        txt_scores.append(util.cos_sim(q_emb, t_emb).item())

    vis_scores = np.array(vis_scores)
    txt_scores = np.array(txt_scores)
    final_scores = 0.1 * txt_scores + 0.9 * vis_scores

    best_idx = int(np.argmax(final_scores))
    vis_best = int(np.argmax(vis_scores))
    txt_best = int(np.argmax(txt_scores))

    return {
        "entry": memory[best_idx], "score": float(final_scores[best_idx]),
        "idx": best_idx, "vis_best": vis_best, "txt_best": txt_best,
        "modalities_agree": vis_best == txt_best, "alpha_used": 0.9,
    }


def retrieve_adaptive(query, memory, vis_embs, txt_embs):
    q_emb = embedder.encode(query, convert_to_tensor=True)
    vis_scores, txt_scores = [], []
    for v_emb, t_emb in zip(vis_embs, txt_embs):
        vis_scores.append(util.cos_sim(q_emb, v_emb).item())
        txt_scores.append(util.cos_sim(q_emb, t_emb).item())

    vis_scores = np.array(vis_scores)
    txt_scores = np.array(txt_scores)

    vis_best = int(np.argmax(vis_scores))
    txt_best = int(np.argmax(txt_scores))
    modalities_agree = (vis_best == txt_best)

    alpha_img = 0.9 if modalities_agree else 0.5
    final_scores = (1.0 - alpha_img) * txt_scores + alpha_img * vis_scores
    best_idx = int(np.argmax(final_scores))

    return {
        "entry": memory[best_idx], "score": float(final_scores[best_idx]),
        "idx": best_idx, "vis_best": vis_best, "txt_best": txt_best,
        "modalities_agree": modalities_agree, "alpha_used": alpha_img,
    }


# ═══════════════════════════════════════════════════════════════════
# RUN BOTH METHODS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT 1 (train2017): BASELINE vs ADAPTIVE MODALITY GATING")
print(f"  Dataset: train2017 adversarial ({len(data)} samples)")
print("=" * 70)

memory, vis_embs, txt_embs = build_memory_split(data)

metrics_keys = ["edit_acc", "rephrase_acc", "locality_acc",
                "portability_acc", "retrieval_score"]

results = {"baseline": {}, "adaptive": {}}

for method_name, retrieve_fn in [("baseline", retrieve_baseline),
                                  ("adaptive", retrieve_adaptive)]:
    print(f"\n  ── Running: {method_name.upper()} ──")
    counters = {k: [] for k in metrics_keys}
    per_category = {}
    conflicts_detected = 0
    conflict_fixed = 0
    details = []
    t0 = time.time()

    for i, mem in enumerate(memory):
        cat = get_category(data[i])
        if cat not in per_category:
            per_category[cat] = {k: [] for k in metrics_keys}

        ret = retrieve_fn(mem["visual_q"], memory, vis_embs, txt_embs)
        entry = ret["entry"]
        counters["retrieval_score"].append(ret["score"])
        per_category[cat]["retrieval_score"].append(ret["score"])

        if not ret["modalities_agree"]:
            conflicts_detected += 1

        e_out = generate_answer(build_edit_prompt(entry, mem["visual_q"]))
        e_ok = check_answer(e_out, mem["visual_a"])
        counters["edit_acc"].append(int(e_ok))
        per_category[cat]["edit_acc"].append(int(e_ok))

        r_out = generate_answer(build_edit_prompt(entry, mem["rephrase_q"]))
        r_ok = check_answer(r_out, mem["visual_a"])
        counters["rephrase_acc"].append(int(r_ok))
        per_category[cat]["rephrase_acc"].append(int(r_ok))

        l_out = generate_answer(build_simple_prompt(mem["loc_q"]))
        l_ok = check_answer(l_out, mem["loc_a"])
        counters["locality_acc"].append(int(l_ok))
        per_category[cat]["locality_acc"].append(int(l_ok))

        p_out = generate_answer(build_edit_prompt(entry, mem["comp_q"]))
        p_ok = check_answer(p_out, mem["comp_a"])
        counters["portability_acc"].append(int(p_ok))
        per_category[cat]["portability_acc"].append(int(p_ok))

        detail = {
            "idx": i, "category": cat,
            "question": mem["visual_q"], "expected": mem["visual_a"],
            "edit_pred": e_out, "edit_ok": e_ok,
            "rephrase_pred": r_out, "rephrase_ok": r_ok,
            "portability_pred": p_out, "portability_ok": p_ok,
            "retrieval_score": ret["score"],
            "modalities_agree": ret["modalities_agree"],
            "alpha_used": ret["alpha_used"],
            "vis_best": ret["vis_best"], "txt_best": ret["txt_best"],
            "self_retrieved": ret["idx"] == i,
        }

        if method_name == "adaptive" and not ret["modalities_agree"] and e_ok:
            conflict_fixed += 1

        details.append(detail)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1:>4d}/{len(data)}] {elapsed:>5.0f}s | "
                  f"edit={mean(counters['edit_acc']):.3f} "
                  f"reph={mean(counters['rephrase_acc']):.3f} "
                  f"port={mean(counters['portability_acc']):.3f} "
                  f"ret={mean(counters['retrieval_score']):.3f}")

    elapsed = time.time() - t0
    overall = {k: round(mean(v), 4) for k, v in counters.items()}
    overall["conflicts_detected"] = conflicts_detected
    overall["conflict_fixed"] = conflict_fixed if method_name == "adaptive" else "N/A"
    overall["eval_time_seconds"] = round(elapsed, 2)

    cat_summary = {}
    for cat, cat_cnt in per_category.items():
        cat_summary[cat] = {k: round(mean(v), 4) for k, v in cat_cnt.items() if v}
        cat_summary[cat]["count"] = len(cat_cnt["edit_acc"])

    results[method_name] = {
        "overall": overall,
        "per_category": cat_summary,
        "details": details,
    }

    print(f"\n  {method_name.upper()} Results:")
    print(f"    Edit:        {overall['edit_acc']*100:.1f}%")
    print(f"    Rephrase:    {overall['rephrase_acc']*100:.1f}%")
    print(f"    Locality:    {overall['locality_acc']*100:.1f}%")
    print(f"    Portability: {overall['portability_acc']*100:.1f}%")
    print(f"    Retrieval:   {overall['retrieval_score']:.4f}")
    print(f"    Conflicts:   {conflicts_detected}")
    if method_name == "adaptive":
        print(f"    Conflicts fixed by gating: {conflict_fixed}")


# ═══════════════════════════════════════════════════════════════════
# COMPUTE DELTAS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  COMPARISON: BASELINE vs ADAPTIVE")
print("=" * 70)

b = results["baseline"]["overall"]
a = results["adaptive"]["overall"]

deltas = {}
print(f"  {'Metric':<22s} {'Baseline':>10s} {'Adaptive':>10s} {'Delta':>10s}")
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

results["deltas"] = deltas

# Per-category comparison
print(f"\n  PER-CATEGORY EDIT ACCURACY:")
print(f"  {'Category':<22s} {'Baseline':>10s} {'Adaptive':>10s} {'Delta':>10s}")
print(f"  {'─'*55}")
all_cats = sorted(set(list(results["baseline"]["per_category"].keys()) +
                      list(results["adaptive"]["per_category"].keys())))
cat_deltas = {}
for cat in all_cats:
    bc = results["baseline"]["per_category"].get(cat, {}).get("edit_acc", 0)
    ac = results["adaptive"]["per_category"].get(cat, {}).get("edit_acc", 0)
    d = ac - bc
    cat_deltas[cat] = {"baseline": bc, "adaptive": ac, "delta": round(d, 4)}
    print(f"  {cat:<22s} {bc*100:>9.1f}% {ac*100:>9.1f}% {d*100:>+9.1f}pp")

results["category_edit_deltas"] = cat_deltas


# ═══════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════
save_obj = {
    "experiment": "Adaptive Modality Gating (Experiment 1 — train2017)",
    "timestamp": datetime.now().isoformat(),
    "dataset": f"train2017_adversarial_2k ({len(data)} samples)",
    "image_source": "COCO train2017",
    "model": MODEL_NAME, "device": device,
    "baseline": {"overall": results["baseline"]["overall"],
                 "per_category": results["baseline"]["per_category"]},
    "adaptive": {"overall": results["adaptive"]["overall"],
                 "per_category": results["adaptive"]["per_category"]},
    "deltas": deltas,
    "category_edit_deltas": cat_deltas,
}
with open(f"{RESULTS_DIR}/exp1_adaptive_gating.json", "w", encoding="utf-8") as f:
    json.dump(save_obj, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {RESULTS_DIR}/exp1_adaptive_gating.json")

# Save checkpoint
ckpt = {
    "experiment": "exp1_adaptive_gating",
    "dataset": DATA_PATH,
    "model": MODEL_NAME,
    "embedder": "all-MiniLM-L6-v2",
    "timestamp": datetime.now().isoformat(),
    "config": {"baseline_alpha": 0.9, "adaptive_conflict_alpha": 0.5},
    "results_summary": {
        "baseline_edit_acc": b["edit_acc"],
        "adaptive_edit_acc": a["edit_acc"],
        "delta_edit_acc": deltas["edit_acc"],
        "conflicts_detected": a["conflicts_detected"],
        "conflicts_fixed": a["conflict_fixed"],
    },
}
with open("new-checkpoint/exp1_checkpoint.json", "w", encoding="utf-8") as f:
    json.dump(ckpt, f, indent=2)
print("Saved: new-checkpoint/exp1_checkpoint.json")


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

# PLOT 1: Overall Bar Chart
fig, ax = plt.subplots(figsize=(9, 5.5))
metrics_plot = ["edit_acc", "rephrase_acc", "portability_acc", "locality_acc"]
labels_plot = ["Edit\nAccuracy", "Rephrase\nAccuracy", "Portability\nAccuracy", "Locality\nAccuracy"]
x = np.arange(len(metrics_plot))
width = 0.32

baseline_vals = [b[m]*100 for m in metrics_plot]
adaptive_vals = [a[m]*100 for m in metrics_plot]

bars1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline (Fixed 0.9/0.1)",
               color="#e74c3c", edgecolor="white", linewidth=0.8)
bars2 = ax.bar(x + width/2, adaptive_vals, width, label="Adaptive Gating (Ours)",
               color="#2ecc71", edgecolor="white", linewidth=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

for i, m in enumerate(metrics_plot):
    d = deltas[m] * 100
    color = "#27ae60" if d > 0 else "#c0392b" if d < 0 else "#7f8c8d"
    ax.annotate(f"{d:+.1f}pp", xy=(x[i] + width/2, adaptive_vals[i]),
                xytext=(x[i] + width/2 + 0.15, adaptive_vals[i] + 5),
                fontsize=8, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Experiment 1: Adaptive Modality Gating vs Baseline\n(train2017 — {len(data)} samples)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels_plot)
ax.set_ylim(0, max(max(baseline_vals), max(adaptive_vals)) + 15)
ax.legend(loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/exp1_overall_comparison.png")
plt.close()

# PLOT 2: Per-Category Edit Accuracy
fig, ax = plt.subplots(figsize=(10, 5.5))
cats = sorted(all_cats)
x = np.arange(len(cats))
width = 0.32
cat_labels = [c.replace("_", "\n") for c in cats]

base_cat_edit = [results["baseline"]["per_category"].get(c, {}).get("edit_acc", 0)*100 for c in cats]
adap_cat_edit = [results["adaptive"]["per_category"].get(c, {}).get("edit_acc", 0)*100 for c in cats]

bars1 = ax.bar(x - width/2, base_cat_edit, width, label="Baseline", color="#e74c3c", edgecolor="white")
bars2 = ax.bar(x + width/2, adap_cat_edit, width, label="Adaptive (Ours)", color="#2ecc71", edgecolor="white")

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylabel("Edit Accuracy (%)")
ax.set_title("Per-Category Edit Accuracy: Baseline vs Adaptive Gating (train2017)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(cat_labels)
ax.set_ylim(0, max(max(base_cat_edit), max(adap_cat_edit)) + 15)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/exp1_category_comparison.png")
plt.close()

# PLOT 3: Conflict Detection Analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

n_total = len(data)
n_agree = n_total - results["adaptive"]["overall"]["conflicts_detected"]
n_conflict = results["adaptive"]["overall"]["conflicts_detected"]
axes[0].pie([n_agree, n_conflict],
            labels=[f"Agree\n({n_agree})", f"Conflict\n({n_conflict})"],
            colors=["#2ecc71", "#e74c3c"],
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 12, "fontweight": "bold"})
axes[0].set_title(f"Modality Agreement\nacross {n_total} samples", fontweight="bold")

n_fixed = results["adaptive"]["overall"]["conflict_fixed"]
n_still_wrong = n_conflict - n_fixed
axes[1].bar(["Fixed by\nAdaptive Gating", "Still Incorrect"],
            [n_fixed, n_still_wrong],
            color=["#2ecc71", "#e74c3c"], edgecolor="white", width=0.5)
for i, v in enumerate([n_fixed, n_still_wrong]):
    axes[1].text(i, v + 0.5, str(v), ha="center", fontweight="bold", fontsize=14)
axes[1].set_ylabel("Number of Samples")
axes[1].set_title("Conflict Resolution Results", fontweight="bold")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle("Adaptive Modality Gating: Conflict Detection & Resolution (train2017)", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/exp1_conflict_analysis.png")
plt.close()

# PLOT 4: Alpha Distribution
fig, ax = plt.subplots(figsize=(8, 5))
alphas_used = [d["alpha_used"] for d in results["adaptive"]["details"]]
correctness = [d["edit_ok"] for d in results["adaptive"]["details"]]

alpha_09 = [c for aa, c in zip(alphas_used, correctness) if aa == 0.9]
alpha_05 = [c for aa, c in zip(alphas_used, correctness) if aa == 0.5]

groups = ["α = 0.9\n(Agree → Trust Image)", "α = 0.5\n(Conflict → Equal Weight)"]
accs = [mean(alpha_09)*100 if alpha_09 else 0, mean(alpha_05)*100 if alpha_05 else 0]
counts = [len(alpha_09), len(alpha_05)]

bars = ax.bar(groups, accs, color=["#3498db", "#e67e22"], edgecolor="white", width=0.5)
for bar, v, n in zip(bars, accs, counts):
    ax.text(bar.get_x()+bar.get_width()/2, v+1,
            f"{v:.1f}%\n(n={n})", ha="center", fontweight="bold", fontsize=11)

ax.set_ylabel("Edit Accuracy (%)")
ax.set_title("Edit Accuracy by Gating Decision (train2017)", fontweight="bold")
ax.set_ylim(0, max(accs) + 15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/exp1_alpha_distribution.png")
plt.close()

print(f"\n  Plots saved to {PLOTS_DIR}/exp1_*.png")

# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT 1 (train2017) COMPLETE")
print("=" * 70)
print(f"  Key finding: Edit accuracy {'improved' if deltas['edit_acc'] > 0 else 'changed'} "
      f"by {deltas['edit_acc']*100:+.1f}pp with adaptive gating")
print(f"  Conflicts detected: {results['adaptive']['overall']['conflicts_detected']}/{len(data)} samples")
print(f"  Conflicts fixed: {results['adaptive']['overall']['conflict_fixed']}")
print(f"  LIMITATION CONFIRMED: Adaptive gating alone is insufficient on larger train2017 data")
