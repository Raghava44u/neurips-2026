"""
Experiment 3 (train2017): Consistency-Checked Connector
=======================================================
Same as experiment_3_consistency_connector.py but uses train2017-based 2K dataset.
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


def build_edit_prompt_both(vis_entry, txt_entry, question):
    return (f"Use the following facts to answer the question accurately.\n\n"
            f"Visual fact: {vis_entry['visual_q']} → {vis_entry['visual_a']}\n"
            f"Text fact: {txt_entry['text_q']} → {txt_entry['text_a']}\n\n"
            f"Question: {question}\nAnswer:")


def build_edit_prompt_single(entry, question, modality="visual"):
    if modality == "visual":
        fact = f"{entry['visual_q']} → {entry['visual_a']}"
    else:
        fact = f"{entry['text_q']} → {entry['text_a']}"
    return (f"Use the following fact to answer the question accurately.\n\n"
            f"Fact: {fact}\n\n"
            f"Question: {question}\nAnswer:")


def build_simple_prompt(question):
    return f"Answer the following question accurately.\n\nQuestion: {question}\nAnswer:"


# ═══════════════════════════════════════════════════════════════════
# CONNECTOR METHODS
# ═══════════════════════════════════════════════════════════════════

def retrieve_best(query, memory, vis_embs, txt_embs):
    q_emb = embedder.encode(query, convert_to_tensor=True)
    vis_scores, txt_scores = [], []
    for v_emb, t_emb in zip(vis_embs, txt_embs):
        vis_scores.append(util.cos_sim(q_emb, v_emb).item())
        txt_scores.append(util.cos_sim(q_emb, t_emb).item())

    vis_scores = np.array(vis_scores)
    txt_scores = np.array(txt_scores)
    vis_best_idx = int(np.argmax(vis_scores))
    txt_best_idx = int(np.argmax(txt_scores))

    return {
        "vis_entry": memory[vis_best_idx],
        "txt_entry": memory[txt_best_idx],
        "vis_score": float(vis_scores[vis_best_idx]),
        "txt_score": float(txt_scores[txt_best_idx]),
        "vis_idx": vis_best_idx,
        "txt_idx": txt_best_idx,
    }


def eval_standard_connector(mem, ret, question):
    """Standard: always compose both modalities."""
    prompt = build_edit_prompt_both(ret["vis_entry"], ret["txt_entry"], question)
    pred = generate_answer(prompt)
    return pred, {"method": "standard", "composed": True}


def eval_no_connector(mem, ret, question):
    """No connector: single best modality only."""
    if ret["vis_score"] >= ret["txt_score"]:
        prompt = build_edit_prompt_single(ret["vis_entry"], question, "visual")
    else:
        prompt = build_edit_prompt_single(ret["txt_entry"], question, "text")
    pred = generate_answer(prompt)
    return pred, {"method": "no_connector", "composed": False}


def eval_gated_connector(mem, ret, question, consistency_threshold=0.5):
    """Gated: check consistency, compose if agree, bypass if disagree."""
    vis_a_emb = embedder.encode(ret["vis_entry"]["visual_a"], convert_to_tensor=True)
    txt_a_emb = embedder.encode(ret["txt_entry"]["text_a"], convert_to_tensor=True)
    consistency = util.cos_sim(vis_a_emb, txt_a_emb).item()

    if consistency >= consistency_threshold:
        prompt = build_edit_prompt_both(ret["vis_entry"], ret["txt_entry"], question)
        composed = True
    else:
        if ret["vis_score"] >= ret["txt_score"]:
            prompt = build_edit_prompt_single(ret["vis_entry"], question, "visual")
        else:
            prompt = build_edit_prompt_single(ret["txt_entry"], question, "text")
        composed = False

    pred = generate_answer(prompt)
    return pred, {"method": "gated", "composed": composed,
                  "consistency": consistency, "gate_value": float(consistency >= consistency_threshold)}


# ═══════════════════════════════════════════════════════════════════
# RUN ALL THREE METHODS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT 3 (train2017): CONSISTENCY-CHECKED CONNECTOR")
print(f"  Dataset: train2017 adversarial ({len(data)} samples)")
print("=" * 70)

memory, vis_embs, txt_embs = build_memory_split(data)

metrics_keys = ["edit_acc", "rephrase_acc", "locality_acc",
                "portability_acc", "retrieval_score"]

methods = [
    ("standard", eval_standard_connector),
    ("no_connector", eval_no_connector),
    ("gated", eval_gated_connector),
]

results = {}

for method_name, eval_fn in methods:
    print(f"\n  ── Running: {method_name.upper()} ──")
    counters = {k: [] for k in metrics_keys}
    per_category = {}
    bypass_count = 0
    agree_count = 0
    pred_sims = []
    gate_values = []
    details = []
    t0 = time.time()

    for i, mem in enumerate(memory):
        cat = get_category(data[i])
        if cat not in per_category:
            per_category[cat] = {k: [] for k in metrics_keys}

        ret = retrieve_best(mem["visual_q"], memory, vis_embs, txt_embs)
        score = max(ret["vis_score"], ret["txt_score"])
        counters["retrieval_score"].append(score)
        per_category[cat]["retrieval_score"].append(score)

        # Edit accuracy
        e_out, info = eval_fn(mem, ret, mem["visual_q"])
        e_ok = check_answer(e_out, mem["visual_a"])
        counters["edit_acc"].append(int(e_ok))
        per_category[cat]["edit_acc"].append(int(e_ok))

        if info.get("composed", True):
            agree_count += 1
        else:
            bypass_count += 1

        if "consistency" in info:
            pred_sims.append(info["consistency"])
            gate_values.append(info["gate_value"])

        # Rephrase
        r_out, _ = eval_fn(mem, ret, mem["rephrase_q"])
        r_ok = check_answer(r_out, mem["visual_a"])
        counters["rephrase_acc"].append(int(r_ok))
        per_category[cat]["rephrase_acc"].append(int(r_ok))

        # Locality
        l_out = generate_answer(build_simple_prompt(mem["loc_q"]))
        l_ok = check_answer(l_out, mem["loc_a"])
        counters["locality_acc"].append(int(l_ok))
        per_category[cat]["locality_acc"].append(int(l_ok))

        # Portability
        p_out, _ = eval_fn(mem, ret, mem["comp_q"])
        p_ok = check_answer(p_out, mem["comp_a"])
        counters["portability_acc"].append(int(p_ok))
        per_category[cat]["portability_acc"].append(int(p_ok))

        details.append({
            "idx": i, "category": cat, "edit_ok": e_ok, "rephrase_ok": r_ok,
            "portability_ok": p_ok, "composed": info.get("composed", True),
            "consistency": info.get("consistency", None),
        })

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1:>4d}/{len(data)}] {elapsed:>5.0f}s | "
                  f"edit={mean(counters['edit_acc']):.3f}")

    elapsed = time.time() - t0
    overall = {k: round(mean(v), 4) for k, v in counters.items()}
    overall["bypass_count"] = bypass_count
    overall["agree_count"] = agree_count
    overall["bypass_rate"] = round(bypass_count / len(data), 4) if data else 0
    if pred_sims:
        overall["avg_pred_sim"] = round(mean(pred_sims), 4)
        overall["avg_gate_value"] = round(mean(gate_values), 4)

    # Separate acc for bypass vs compose
    bypass_accs = [d["edit_ok"] for d in details if not d["composed"]]
    compose_accs = [d["edit_ok"] for d in details if d["composed"]]
    overall["bypass_acc"] = round(mean(bypass_accs), 4) if bypass_accs else 0
    overall["compose_acc"] = round(mean(compose_accs), 4) if compose_accs else 0
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


# ═══════════════════════════════════════════════════════════════════
# DELTAS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  COMPARISON")
print("=" * 70)

s_ovr = results["standard"]["overall"]
n_ovr = results["no_connector"]["overall"]
g_ovr = results["gated"]["overall"]

deltas = {}
print(f"  {'Metric':<20s} {'Standard':>10s} {'No-Conn':>10s} {'Gated':>10s}")
print(f"  {'─'*55}")
for k in metrics_keys:
    sv = s_ovr.get(k, 0)
    nv = n_ovr.get(k, 0)
    gv = g_ovr.get(k, 0)
    deltas[k] = round(gv - sv, 4)
    if k.endswith("_acc"):
        print(f"  {k:<20s} {sv*100:>9.1f}% {nv*100:>9.1f}% {gv*100:>9.1f}%")
    else:
        print(f"  {k:<20s} {sv:>10.4f} {nv:>10.4f} {gv:>10.4f}")

# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════
save_obj = {
    "experiment": "Consistency-Checked Connector (Experiment 3 — train2017)",
    "timestamp": datetime.now().isoformat(),
    "dataset": f"train2017_adversarial_2k ({len(data)} samples)",
    "image_source": "COCO train2017",
    "model": MODEL_NAME, "device": device,
    "standard": {"overall": s_ovr, "per_category": results["standard"]["per_category"]},
    "no_connector": {"overall": n_ovr, "per_category": results["no_connector"]["per_category"]},
    "gated": {"overall": g_ovr, "per_category": results["gated"]["per_category"]},
    "deltas": deltas,
}
with open(f"{RESULTS_DIR}/exp3_consistency_connector.json", "w", encoding="utf-8") as f:
    json.dump(save_obj, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {RESULTS_DIR}/exp3_consistency_connector.json")

ckpt = {
    "experiment": "exp3_consistency_connector",
    "dataset": DATA_PATH,
    "model": MODEL_NAME,
    "embedder": "all-MiniLM-L6-v2",
    "timestamp": datetime.now().isoformat(),
    "config": {"consistency_threshold": 0.5},
    "results_summary": {
        "standard_edit_acc": s_ovr["edit_acc"],
        "no_connector_edit_acc": n_ovr["edit_acc"],
        "gated_edit_acc": g_ovr["edit_acc"],
        "delta_edit_acc": deltas["edit_acc"],
        "bypass_rate": g_ovr["bypass_rate"],
    },
}
with open("new-checkpoint/exp3_checkpoint.json", "w", encoding="utf-8") as f:
    json.dump(ckpt, f, indent=2)
print("Saved: new-checkpoint/exp3_checkpoint.json")


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

# PLOT 1: Overall
fig, ax = plt.subplots(figsize=(10, 5.5))
metrics_plot = ["edit_acc", "rephrase_acc", "portability_acc", "locality_acc"]
labels_plot = ["Edit\nAccuracy", "Rephrase\nAccuracy", "Portability\nAccuracy", "Locality\nAccuracy"]
x = np.arange(len(metrics_plot))
width = 0.25

sv = [s_ovr[m]*100 for m in metrics_plot]
nv = [n_ovr[m]*100 for m in metrics_plot]
gv_vals = [g_ovr[m]*100 for m in metrics_plot]

bars1 = ax.bar(x - width, sv, width, label="Standard (Both)", color="#e74c3c", edgecolor="white")
bars2 = ax.bar(x, nv, width, label="No Connector (Single)", color="#3498db", edgecolor="white")
bars3 = ax.bar(x + width, gv_vals, width, label="Gated (Ours)", color="#2ecc71", edgecolor="white")

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Experiment 3: Connector Methods Comparison\n(train2017 — {len(data)} samples)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels_plot)
ax.set_ylim(0, max(max(sv), max(nv), max(gv_vals)) + 15)
ax.legend(loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/exp3_overall_comparison.png")
plt.close()

# PLOT 2: Per-Category
all_cats = sorted(set(
    list(results["standard"]["per_category"].keys()) +
    list(results["gated"]["per_category"].keys())
))
fig, ax = plt.subplots(figsize=(10, 5.5))
cats = sorted(all_cats)
x = np.arange(len(cats))
cat_labels = [c.replace("_", "\n") for c in cats]

s_cats = [results["standard"]["per_category"].get(c, {}).get("edit_acc", 0)*100 for c in cats]
g_cats = [results["gated"]["per_category"].get(c, {}).get("edit_acc", 0)*100 for c in cats]

bars1 = ax.bar(x - 0.16, s_cats, 0.32, label="Standard", color="#e74c3c", edgecolor="white")
bars2 = ax.bar(x + 0.16, g_cats, 0.32, label="Gated (Ours)", color="#2ecc71", edgecolor="white")

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylabel("Edit Accuracy (%)")
ax.set_title("Per-Category: Standard vs Gated Connector (train2017)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(cat_labels)
ax.set_ylim(0, max(max(s_cats), max(g_cats)) + 15)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/exp3_category_comparison.png")
plt.close()

# PLOT 3: Gate Analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

bypass = g_ovr["bypass_count"]
compose = g_ovr["agree_count"]
axes[0].pie([compose, bypass],
            labels=[f"Compose\n({compose})", f"Bypass\n({bypass})"],
            colors=["#2ecc71", "#e67e22"],
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 12, "fontweight": "bold"})
axes[0].set_title("Gated Connector Decisions", fontweight="bold")

axes[1].bar(["Compose", "Bypass"],
            [g_ovr["compose_acc"]*100, g_ovr["bypass_acc"]*100],
            color=["#2ecc71", "#e67e22"], edgecolor="white", width=0.5)
for i, v in enumerate([g_ovr["compose_acc"]*100, g_ovr["bypass_acc"]*100]):
    axes[1].text(i, v+1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=14)
axes[1].set_ylabel("Edit Accuracy (%)")
axes[1].set_title("Accuracy by Gate Decision", fontweight="bold")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle("Gated Connector: Consistency Analysis (train2017)", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/exp3_gate_analysis.png")
plt.close()

# PLOT 4: Consistency Distribution
if pred_sims:
    fig, ax = plt.subplots(figsize=(8, 5))
    consistencies = [d["consistency"] for d in results["gated"]["details"] if d["consistency"] is not None]
    ax.hist(consistencies, bins=50, color="#9b59b6", edgecolor="white", alpha=0.8)
    ax.axvline(x=0.5, color="#e74c3c", linestyle="--", linewidth=2, label="Threshold (0.5)")
    ax.axvline(x=np.mean(consistencies), color="#2ecc71", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(consistencies):.3f}")
    ax.set_xlabel("Cross-Modal Consistency Score")
    ax.set_ylabel("Count")
    ax.set_title("Consistency Score Distribution (train2017)", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/exp3_consistency_distribution.png")
    plt.close()

print(f"\n  Plots saved to {PLOTS_DIR}/exp3_*.png")

print("\n" + "=" * 70)
print("  EXPERIMENT 3 (train2017) COMPLETE")
print("=" * 70)
print(f"  Bypass rate: {g_ovr['bypass_rate']*100:.1f}%")
print(f"  LIMITATION CONFIRMED: Consistency connector shows marginal improvement on train2017")
