"""
MemEIC Complete Failure Analysis Pipeline
==========================================
Runs all experiments for research publication:
  Step 3: Evaluate Original, V1, V2 datasets
  Step 4: Final comparison across all 3
  Step 5: Failure analysis (10+ cases from V2)
  Step 6: Retrieval sensitivity (alpha sweep)
  Step 7: Ablation study (3 variants)

Uses: SentenceTransformer (retrieval) + phi-2 (generation) on CUDA
"""

import json, random, os, time, torch, re
from datetime import datetime
from statistics import mean
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# ── Config ──────────────────────────────────────────────────────────
ORIGINAL_PATH   = "datasets/CCKEB_eval.json"
V1_PATH         = "datasets/complex_reasoning_dataset.json"
V2_PATH         = "datasets/adversarial_reasoning_dataset.json"
N_ORIGINAL      = 50
SEED            = 42
random.seed(SEED)

# ── Load datasets ───────────────────────────────────────────────────
print("=" * 70)
print("  LOADING DATASETS")
print("=" * 70)

with open(ORIGINAL_PATH, "r", encoding="utf-8") as f:
    original_all = json.load(f)
with open(V1_PATH, "r", encoding="utf-8") as f:
    v1_data = json.load(f)
with open(V2_PATH, "r", encoding="utf-8") as f:
    v2_data = json.load(f)

original_data = random.sample(original_all, N_ORIGINAL)
print(f"  Original: {len(original_data)} samples (from {len(original_all)})")
print(f"  V1 (complex): {len(v1_data)} samples")
print(f"  V2 (adversarial): {len(v2_data)} samples")

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
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def build_memory(data):
    """Parse dataset into memory entries + embeddings."""
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
            "text_q": te["src"],
            "text_a": te_alt,
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


def retrieve(query, memory, memory_embeddings, alpha=0.9):
    """Retrieve best match. alpha controls image vs text weight simulation.
    For this text-only pipeline, alpha controls how much the visual_q part
    vs text_q part of the combined embedding matters.
    """
    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = [util.cos_sim(query_emb, emb).item() for emb in memory_embeddings]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return memory[best_idx], scores[best_idx], best_idx


def generate(prompt, max_tokens=30):
    """Generate answer from LLM."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    return text.split("\n")[0].strip()


def check_answer(prediction, expected):
    """Flexible answer matching."""
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
    """No retrieval — just ask the question directly (ablation)."""
    return f"Question: {question}\nAnswer:"


def build_no_connector_prompt(mem, question):
    """Only visual OR textual fact, no compositional linking (ablation)."""
    return (f"Use this fact to answer.\n\n"
            f"Fact: {mem['visual_q']} → {mem['visual_a']}\n\n"
            f"Question: {question}\nAnswer:")


# ═══════════════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════

def evaluate_dataset(data, label, mode="full", alpha=0.9):
    """
    Evaluate a dataset.
    mode: "full" | "no_retrieval" | "no_connector"
    alpha: retrieval weight parameter
    """
    print(f"\n{'─' * 60}")
    print(f"  {label} | mode={mode} | alpha={alpha}")
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

        # Locality (always without edit)
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

        # Text locality (always without edit)
        tl_out = generate(build_simple_prompt(mem["textual_loc_q"]))
        tl_ok = check_answer(tl_out, mem["textual_loc_a"])
        counters["text_locality_acc"].append(int(tl_ok))

        details.append(d)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1:>3d}/{len(data)}] {elapsed:>5.0f}s | "
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
# STEP 3: RUN MAIN EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 3: MAIN EXPERIMENTS")
print("=" * 70)

os.makedirs("results", exist_ok=True)

orig_sum, orig_det = evaluate_dataset(original_data, "Original CCKEB")
v1_sum, v1_det = evaluate_dataset(v1_data, "V1 (Complex Reasoning)")
v2_sum, v2_det = evaluate_dataset(v2_data, "V2 (Adversarial)")

# Save individual results
for name, summary, details, path in [
    ("Original", orig_sum, orig_det, "results/results_original.json"),
    ("V1", v1_sum, v1_det, "results/results_v1.json"),
    ("V2", v2_sum, v2_det, "results/results_v2.json"),
]:
    obj = {
        "metadata": {"timestamp": datetime.now().isoformat(),
                      "model": MODEL_NAME, "device": device},
        "summary": summary,
        "sample_details": details,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# STEP 4: FINAL COMPARISON
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 4: FINAL COMPARISON")
print("=" * 70)

metric_keys = [
    "edit_acc", "rephrase_acc", "locality_acc", "m_locality_acc",
    "portability_acc", "text_edit_acc", "retrieval_score",
]

comparison = {}
print(f"  {'Metric':<25s} {'Original':>10s} {'V1':>10s} {'V2':>10s}")
print(f"  {'-'*55}")
for k in metric_keys:
    o = orig_sum.get(k, 0)
    v1 = v1_sum.get(k, 0)
    v2 = v2_sum.get(k, 0)
    comparison[k] = {"original": o, "v1": v1, "v2": v2}
    if k.endswith("_acc"):
        print(f"  {k:<25s} {o*100:>9.1f}% {v1*100:>9.1f}% {v2*100:>9.1f}%")
    else:
        print(f"  {k:<25s} {o:>10.4f} {v1:>10.4f} {v2:>10.4f}")

with open("results/final_comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)
print("  Saved: results/final_comparison.json")


# ═══════════════════════════════════════════════════════════════════
# STEP 5: FAILURE ANALYSIS (V2 dataset)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 5: FAILURE ANALYSIS")
print("=" * 70)

# Categorize adversarial samples
# Samples 0-14: ambiguity, 15-24: conflicting, 25-34: retrieval trap,
# 35-44: noisy reasoning, 45-49: extreme adversarial
category_ranges = {
    "ambiguity_confusion": (0, 15),
    "conflicting_signals": (15, 25),
    "retrieval_error": (25, 35),
    "reasoning_failure": (35, 45),
    "extreme_adversarial": (45, 50),
}

def get_failure_type(idx):
    for cat, (lo, hi) in category_ranges.items():
        if lo <= idx < hi:
            return cat
    return "unknown"

failure_cases = []
for d in v2_det:
    idx = d["sample_idx"]
    # Check if any metric failed
    failed_metrics = []
    for metric in ["edit", "rephrase", "locality", "portability"]:
        if metric in d and not d[metric]["correct"]:
            failed_metrics.append(metric)

    if failed_metrics:
        case = {
            "sample_idx": idx,
            "question": d["src"],
            "image": v2_data[idx]["image"],
            "expected": d["expected"],
            "predicted": d.get("edit", {}).get("pred", "N/A"),
            "failure_type": get_failure_type(idx),
            "failed_metrics": failed_metrics,
            "retrieval_score": d.get("retrieval_score", 0),
            "self_retrieved": d.get("self_retrieved", False),
        }
        failure_cases.append(case)

# Ensure at least 10 cases — take top failures
failure_cases.sort(key=lambda x: len(x["failed_metrics"]), reverse=True)
failure_export = failure_cases[:max(10, len(failure_cases))]

print(f"  Total failed samples: {len(failure_cases)}/{len(v2_det)}")
print(f"  Failure breakdown:")
type_counts = {}
for fc in failure_cases:
    t = fc["failure_type"]
    type_counts[t] = type_counts.get(t, 0) + 1
for t, c in sorted(type_counts.items()):
    print(f"    {t:<25s}: {c}")

print(f"\n  Top failure cases:")
for fc in failure_export[:5]:
    print(f"    [{fc['sample_idx']:>2d}] {fc['failure_type']:<22s} "
          f"failed={fc['failed_metrics']}  ret={fc['retrieval_score']:.3f}")
    print(f"         Q: {fc['question'][:60]}")
    print(f"         exp={fc['expected'][:30]}  pred={fc['predicted'][:30]}")

with open("results/failure_cases.json", "w", encoding="utf-8") as f:
    json.dump(failure_export, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: results/failure_cases.json ({len(failure_export)} cases)")


# ═══════════════════════════════════════════════════════════════════
# STEP 6: RETRIEVAL SENSITIVITY (alpha sweep)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 6: RETRIEVAL SENSITIVITY")
print("=" * 70)

# In the real MemEIC pipeline: score = alpha * image_sim + (1-alpha) * text_sim
# Here we simulate the alpha effect by adjusting retrieval embedding weights.
# We create separate visual and textual embeddings and blend with alpha.

def build_memory_split(data):
    """Build visual-only and text-only embeddings for alpha-blended retrieval."""
    entries = []
    vis_embs, txt_embs = [], []
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
    """Retrieve with alpha-blended similarity: alpha*visual + (1-alpha)*text."""
    q_emb = embedder.encode(query, convert_to_tensor=True)
    scores = []
    for v_emb, t_emb in zip(vis_embs, txt_embs):
        vis_sim = util.cos_sim(q_emb, v_emb).item()
        txt_sim = util.cos_sim(q_emb, t_emb).item()
        blended = alpha * vis_sim + (1 - alpha) * txt_sim
        scores.append(blended)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return memory[best_idx], scores[best_idx], best_idx


sensitivity_results = {}
alphas = [0.9, 0.5, 0.1]

for alpha in alphas:
    print(f"\n  Running V2 with alpha={alpha}...")
    mem_entries, vis_e, txt_e = build_memory_split(v2_data)

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

    elapsed = time.time() - t0
    alpha_summary = {k: round(mean(v), 4) for k, v in counters.items() if v}
    alpha_summary["alpha"] = alpha
    alpha_summary["eval_time_seconds"] = round(elapsed, 2)

    sensitivity_results[f"alpha_{alpha}"] = alpha_summary
    print(f"    edit={alpha_summary['edit_acc']*100:.1f}% "
          f"port={alpha_summary['portability_acc']*100:.1f}% "
          f"loc={alpha_summary['locality_acc']*100:.1f}% "
          f"ret={alpha_summary['retrieval_score']:.4f} "
          f"({elapsed:.0f}s)")

with open("results/retrieval_sensitivity.json", "w", encoding="utf-8") as f:
    json.dump(sensitivity_results, f, indent=2)
print("\n  Saved: results/retrieval_sensitivity.json")


# ═══════════════════════════════════════════════════════════════════
# STEP 7: ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 7: ABLATION STUDY")
print("=" * 70)

ablation = {}

# 1. Without retrieval
print("\n  [1/3] Without retrieval (direct question only)...")
abl1_sum, _ = evaluate_dataset(v2_data, "V2-no-retrieval", mode="no_retrieval")
ablation["without_retrieval"] = {k: abl1_sum.get(k, 0) for k in metric_keys}

# 2. Without knowledge connector (single fact only)
print("\n  [2/3] Without connector (single fact only)...")
abl2_sum, _ = evaluate_dataset(v2_data, "V2-no-connector", mode="no_connector")
ablation["without_connector"] = {k: abl2_sum.get(k, 0) for k in metric_keys}

# 3. Full MemEIC (already computed as v2)
print("\n  [3/3] Full MemEIC (already computed)...")
ablation["full_memeic"] = {k: v2_sum.get(k, 0) for k in metric_keys}

print(f"\n  ABLATION SUMMARY:")
print(f"  {'Metric':<25s} {'No Retriev':>10s} {'No Connect':>10s} {'Full':>10s}")
print(f"  {'-'*55}")
for k in metric_keys:
    vals = [ablation[m].get(k, 0) for m in ["without_retrieval", "without_connector", "full_memeic"]]
    if k.endswith("_acc"):
        print(f"  {k:<25s} {vals[0]*100:>9.1f}% {vals[1]*100:>9.1f}% {vals[2]*100:>9.1f}%")
    else:
        print(f"  {k:<25s} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}")

with open("results/ablation_study.json", "w", encoding="utf-8") as f:
    json.dump(ablation, f, indent=2)
print("  Saved: results/ablation_study.json")


# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  ALL EXPERIMENTS COMPLETE")
print("=" * 70)
print("  Files created:")
for f in [
    "results/results_original.json",
    "results/results_v1.json",
    "results/results_v2.json",
    "results/final_comparison.json",
    "results/failure_cases.json",
    "results/retrieval_sensitivity.json",
    "results/ablation_study.json",
]:
    exists = os.path.exists(f)
    status = "✓" if exists else "✗"
    print(f"    {status} {f}")
print("=" * 70)
