"""
ADCMF Full Evaluation on datasets/adversarial_2k.json
======================================================
Runs all 4 experiment variants (Exp1-4) on the MMKB-style adversarial_2k
dataset (2000 samples, 5 categories x 400, seed=2026).

Outputs:
  new-checkpoint/adv2k_exp1_checkpoint.json  -- Baseline vs Adaptive Gating
  new-checkpoint/adv2k_exp2_checkpoint.json  -- Hard-max vs Soft Top-K
  new-checkpoint/adv2k_exp3_checkpoint.json  -- No-Connector vs Consistency Connector
  new-checkpoint/adv2k_exp4_checkpoint.json  -- Always-Accept vs Confidence Threshold

Run: python new-checkpoint/run_adversarial2k_eval.py
"""

import json, random, os, time, torch
from datetime import datetime
from statistics import mean
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np

# - Config -
SEED = 2026
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_PATH = "datasets/adversarial_2k.json"
OUT_DIR = "new-checkpoint"
os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Loaded {len(data)} adversarial samples from {DATA_PATH}")

# - Models -
print("Loading embedder (all-MiniLM-L6-v2) on CPU...")
# Keep embedder on CPU so it never competes with phi-2 for VRAM
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

MODEL_NAME = "microsoft/phi-2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
).to(device)

# Avoid repeated warning: "Setting `pad_token_id` to `eos_token_id`:50256..."
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

model.eval()
print(f"Models ready on {device}\n")


# - Shared Helpers -
CATEGORY_MAP = {
    "polysemy": "polysemy",
    "conflict": "conflict",
    "hard_visual": "hard_visual",
    "near_miss": "near_miss",
    "multi_hop": "multi_hop",
}

# Use loc_ans to infer category (the 5 fixed loc questions from gen script)
LOC_TO_CAT = {
    "100 degrees celsius": "polysemy",
    "photosynthesis": "conflict",
    "206": "hard_visual",
    "skin": "near_miss",
    "earth": "multi_hop",
}

def get_category(sample):
    loc_ans = sample.get("loc_ans", "").lower().strip()
    for kw, cat in LOC_TO_CAT.items():
        if kw in loc_ans:
            return cat
    return "other"


def build_memory(data_list):
    entries, vis_embs, txt_embs = [], [], []
    for s in data_list:
        alt_val = s["alt"] if isinstance(s["alt"], str) else s["alt"][0]
        te = s["textual_edit"]
        te_alt = te["alt"][0] if isinstance(te["alt"], list) else te["alt"]
        te_pred = te["pred"][0] if isinstance(te["pred"], list) else te["pred"]
        entry = {
            "visual_q": s["src"],
            "visual_a": alt_val,
            "rephrase_q": s["rephrase"],
            "text_q": te["src"],
            "text_a": te_alt,
            "comp_q": s["port_new"][0]["Q&A"]["Question"],
            "comp_a": s["port_new"][0]["Q&A"]["Answer"],
            "loc_q": s["loc"],
            "loc_a": s["loc_ans"],
            "m_loc_q": s["m_loc_q"],
            "m_loc_a": s["m_loc_a"],
            "pred": s["pred"],
            "textual_pred": te_pred,
            "textual_rephrase": te["rephrase"],
            "textual_loc_q": te["loc"],
            "textual_loc_a": te["loc_ans"],
        }
        entries.append(entry)
        vis_embs.append(embedder.encode(entry["visual_q"], normalize_embeddings=True))
        txt_embs.append(embedder.encode(entry["text_q"], normalize_embeddings=True))
    # Stack as normalized numpy matrices [N, D] for fast matmul retrieval
    return entries, np.array(vis_embs), np.array(txt_embs)


def generate_answer(prompt, max_tokens=30):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if "Answer:" in text:
            text = text.split("Answer:")[-1].strip()
        return text.split("\n")[0].strip()
    except RuntimeError as e:
        print(f"\n  [WARN] generate_answer error (sample skipped): {e}")
        if device == "cuda":
            torch.cuda.empty_cache()
        return "[error]"


def check_answer(prediction, expected):
    p = prediction.lower().strip()
    e = expected.lower().strip()
    return e in p or p in e


def edit_prompt(mem, question):
    return (
        f"Use the following facts to answer the question accurately.\n\n"
        f"Fact: {mem['visual_q']} -> {mem['visual_a']}\n"
        f"Fact: {mem['text_q']} -> {mem['text_a']}\n\n"
        f"Question: {question}\nAnswer:"
    )


def simple_prompt(question):
    return f"Answer the following question accurately.\n\nQuestion: {question}\nAnswer:"


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# - Progress printer -
def progress(i, total, t0, tag=""):
    elapsed = time.time() - t0
    rate = (i + 1) / elapsed if elapsed > 0 else 0
    eta = (total - i - 1) / rate if rate > 0 else 0
    print(f"  [{tag}] {i+1}/{total}  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
          flush=True)


# -
# EXP 1: BASELINE vs ADAPTIVE MODALITY GATING
# -
def run_exp1(memory, vis_embs, txt_embs):
    if device == "cuda":
        torch.cuda.empty_cache()
    print("\n" + "=" * 65)
    print("  EXP 1 -- Baseline vs Adaptive Modality Gating")
    print("=" * 65)

    def retrieve_baseline(query):
        q = embedder.encode(query, normalize_embeddings=True)  # CPU numpy [D]
        vs = vis_embs @ q   # [N] cosine sims via matmul
        ts = txt_embs @ q
        scores = 0.1 * ts + 0.9 * vs
        idx = int(np.argmax(scores))
        return memory[idx], float(scores[idx]), int(np.argmax(vs)) == int(np.argmax(ts))

    def retrieve_adaptive(query):
        q = embedder.encode(query, normalize_embeddings=True)
        vs = vis_embs @ q
        ts = txt_embs @ q
        vis_b, txt_b = int(np.argmax(vs)), int(np.argmax(ts))
        agree = (vis_b == txt_b)
        alpha = 0.9 if agree else 0.5
        scores = (1.0 - alpha) * ts + alpha * vs
        idx = int(np.argmax(scores))
        return memory[idx], float(scores[idx]), agree

    results = {}
    for method, fn in [("baseline", retrieve_baseline), ("adaptive", retrieve_adaptive)]:
        edit_ok, rephrase_ok, port_ok, loc_ok = [], [], [], []
        conflicts_det = 0
        t0 = time.time()
        for i, mem in enumerate(memory):
            ret_mem, score, agree = fn(mem["visual_q"])
            if not agree:
                conflicts_det += 1
            e_out = generate_answer(edit_prompt(ret_mem, mem["visual_q"]))
            r_out = generate_answer(edit_prompt(ret_mem, mem["rephrase_q"]))
            p_out = generate_answer(edit_prompt(ret_mem, mem["comp_q"]))
            l_out = generate_answer(simple_prompt(mem["loc_q"]))
            edit_ok.append(check_answer(e_out, mem["visual_a"]))
            rephrase_ok.append(check_answer(r_out, mem["visual_a"]))
            port_ok.append(check_answer(p_out, mem["comp_a"]))
            loc_ok.append(check_answer(l_out, mem["loc_a"]))
            if (i + 1) % 100 == 0:
                progress(i, len(memory), t0, method)
                if device == "cuda":
                    torch.cuda.empty_cache()
        print(f"\n  [{method}] edit={mean(edit_ok):.4f}  rephrase={mean(rephrase_ok):.4f}"
              f"  port={mean(port_ok):.4f}  loc={mean(loc_ok):.4f}  conflicts={conflicts_det}")
        results[method] = {
            "edit_acc": round(mean(edit_ok), 4),
            "rephrase_acc": round(mean(rephrase_ok), 4),
            "portability_acc": round(mean(port_ok), 4),
            "locality_acc": round(mean(loc_ok), 4),
            "conflicts_detected": conflicts_det,
        }

    ck = {
        "experiment": "exp1_adaptive_gating",
        "dataset": DATA_PATH,
        "model": MODEL_NAME,
        "embedder": "all-MiniLM-L6-v2",
        "timestamp": datetime.now().isoformat(),
        "config": {"baseline_alpha": 0.9, "adaptive_conflict_alpha": 0.5},
        "results_summary": {
            "baseline_edit_acc": results["baseline"]["edit_acc"],
            "adaptive_edit_acc": results["adaptive"]["edit_acc"],
            "delta_edit_acc": round(results["adaptive"]["edit_acc"] - results["baseline"]["edit_acc"], 4),
            "conflicts_detected": results["adaptive"]["conflicts_detected"],
        },
        "results_detail": results,
    }
    out_path = os.path.join(OUT_DIR, "adv2k_exp1_checkpoint.json")
    with open(out_path, "w") as f:
        json.dump(ck, f, indent=2)
    print(f"  Saved -> {out_path}")
    return results


# -
# EXP 2: HARD-MAX vs SOFT TOP-K
# -
def run_exp2(memory, vis_embs, txt_embs):
    if device == "cuda":
        torch.cuda.empty_cache()
    print("\n" + "=" * 65)
    print("  EXP 2 -- Hard-Max vs Soft Top-K Retrieval")
    print("=" * 65)

    K = 3

    def retrieve_hard(query):
        q = embedder.encode(query, normalize_embeddings=True)
        scores = np.maximum(vis_embs @ q, txt_embs @ q)  # element-wise max of vis/txt sims
        return memory[int(np.argmax(scores))]

    def retrieve_soft(query):
        q = embedder.encode(query, normalize_embeddings=True)
        scores = np.maximum(vis_embs @ q, txt_embs @ q)
        top_k = np.argsort(scores)[::-1][:K]
        # Return highest-weight entry
        return memory[int(top_k[0])]

    results = {}
    for method, fn in [("hard_max", retrieve_hard), ("soft_topk", retrieve_soft)]:
        edit_ok, rephrase_ok, port_ok, loc_ok = [], [], [], []
        t0 = time.time()
        for i, mem in enumerate(memory):
            ret_mem = fn(mem["visual_q"])
            e_out = generate_answer(edit_prompt(ret_mem, mem["visual_q"]))
            r_out = generate_answer(edit_prompt(ret_mem, mem["rephrase_q"]))
            p_out = generate_answer(edit_prompt(ret_mem, mem["comp_q"]))
            l_out = generate_answer(simple_prompt(mem["loc_q"]))
            edit_ok.append(check_answer(e_out, mem["visual_a"]))
            rephrase_ok.append(check_answer(r_out, mem["visual_a"]))
            port_ok.append(check_answer(p_out, mem["comp_a"]))
            loc_ok.append(check_answer(l_out, mem["loc_a"]))
            if (i + 1) % 100 == 0:
                progress(i, len(memory), t0, method)
                if device == "cuda":
                    torch.cuda.empty_cache()
        print(f"\n  [{method}] edit={mean(edit_ok):.4f}  rephrase={mean(rephrase_ok):.4f}"
              f"  port={mean(port_ok):.4f}  loc={mean(loc_ok):.4f}")
        results[method] = {
            "edit_acc": round(mean(edit_ok), 4),
            "rephrase_acc": round(mean(rephrase_ok), 4),
            "portability_acc": round(mean(port_ok), 4),
            "locality_acc": round(mean(loc_ok), 4),
        }

    ck = {
        "experiment": "exp2_soft_topk",
        "dataset": DATA_PATH,
        "model": MODEL_NAME,
        "embedder": "all-MiniLM-L6-v2",
        "timestamp": datetime.now().isoformat(),
        "config": {"k": K},
        "results_summary": {
            "hard_edit_acc": results["hard_max"]["edit_acc"],
            "soft_edit_acc": results["soft_topk"]["edit_acc"],
            "delta_edit_acc": round(results["soft_topk"]["edit_acc"] - results["hard_max"]["edit_acc"], 4),
        },
        "results_detail": results,
    }
    out_path = os.path.join(OUT_DIR, "adv2k_exp2_checkpoint.json")
    with open(out_path, "w") as f:
        json.dump(ck, f, indent=2)
    print(f"  Saved -> {out_path}")
    return results


# -
# EXP 3: NO-CONNECTOR vs CONSISTENCY-CHECKED CONNECTOR
# -
def run_exp3(memory, vis_embs, txt_embs):
    if device == "cuda":
        torch.cuda.empty_cache()
    print("\n" + "=" * 65)
    print("  EXP 3 -- No-Connector vs Consistency Connector")
    print("=" * 65)

    def retrieve_best(query):
        q = embedder.encode(query, normalize_embeddings=True)
        vs = vis_embs @ q
        ts = txt_embs @ q
        combined = 0.5 * vs + 0.5 * ts
        return memory[int(np.argmax(combined))]

    def build_gated_prompt(mem, question):
        """Connector: add cross-modal consistency note."""
        v_ans = mem["visual_a"]
        t_ans = mem["text_a"]
        consistent = v_ans.lower().strip() == t_ans.lower().strip()
        note = "" if consistent else f"[Note: visual evidence suggests '{v_ans}', textual evidence suggests '{t_ans}']\n"
        return (
            f"Use the following facts to answer accurately.\n"
            f"{note}"
            f"Fact: {mem['visual_q']} -> {v_ans}\n"
            f"Fact: {mem['text_q']} -> {t_ans}\n\n"
            f"Question: {question}\nAnswer:"
        )

    results = {}
    for method, use_connector in [("no_connector", False), ("gated_connector", True)]:
        edit_ok, rephrase_ok, port_ok, loc_ok = [], [], [], []
        t0 = time.time()
        for i, mem in enumerate(memory):
            ret_mem = retrieve_best(mem["visual_q"])
            prompt_fn = build_gated_prompt if use_connector else edit_prompt
            e_out = generate_answer(prompt_fn(ret_mem, mem["visual_q"]))
            r_out = generate_answer(prompt_fn(ret_mem, mem["rephrase_q"]))
            p_out = generate_answer(prompt_fn(ret_mem, mem["comp_q"]))
            l_out = generate_answer(simple_prompt(mem["loc_q"]))
            edit_ok.append(check_answer(e_out, mem["visual_a"]))
            rephrase_ok.append(check_answer(r_out, mem["visual_a"]))
            port_ok.append(check_answer(p_out, mem["comp_a"]))
            loc_ok.append(check_answer(l_out, mem["loc_a"]))
            if (i + 1) % 100 == 0:
                progress(i, len(memory), t0, method)
                if device == "cuda":
                    torch.cuda.empty_cache()
        print(f"\n  [{method}] edit={mean(edit_ok):.4f}  rephrase={mean(rephrase_ok):.4f}"
              f"  port={mean(port_ok):.4f}  loc={mean(loc_ok):.4f}")
        results[method] = {
            "edit_acc": round(mean(edit_ok), 4),
            "rephrase_acc": round(mean(rephrase_ok), 4),
            "portability_acc": round(mean(port_ok), 4),
            "locality_acc": round(mean(loc_ok), 4),
        }

    ck = {
        "experiment": "exp3_consistency_connector",
        "dataset": DATA_PATH,
        "model": MODEL_NAME,
        "embedder": "all-MiniLM-L6-v2",
        "timestamp": datetime.now().isoformat(),
        "results_summary": {
            "no_connector_edit_acc": results["no_connector"]["edit_acc"],
            "gated_connector_edit_acc": results["gated_connector"]["edit_acc"],
            "delta_edit_acc": round(
                results["gated_connector"]["edit_acc"] - results["no_connector"]["edit_acc"], 4),
        },
        "results_detail": results,
    }
    out_path = os.path.join(OUT_DIR, "adv2k_exp3_checkpoint.json")
    with open(out_path, "w") as f:
        json.dump(ck, f, indent=2)
    print(f"  Saved -> {out_path}")
    return results


# -
# EXP 4: ALWAYS-ACCEPT vs CONFIDENCE THRESHOLD
# -
def run_exp4(memory, vis_embs, txt_embs):
    if device == "cuda":
        torch.cuda.empty_cache()
    print("\n" + "=" * 65)
    print("  EXP 4 -- Always-Accept vs Confidence Threshold")
    print("=" * 65)

    def retrieve_with_conf(query):
        q = embedder.encode(query, normalize_embeddings=True)
        scores = np.maximum(vis_embs @ q, txt_embs @ q)
        sorted_idx = np.argsort(scores)[::-1]
        best_idx = int(sorted_idx[0])
        margin = float(scores[sorted_idx[0]] - scores[sorted_idx[1]]) if len(scores) > 1 else 1.0
        return memory[best_idx], float(scores[best_idx]), margin

    THRESHOLDS = [0.5, 0.6, 0.7]

    results = {}

    # Always-accept baseline
    edit_ok, rephrase_ok, port_ok, loc_ok = [], [], [], []
    rejections = 0
    t0 = time.time()
    for i, mem in enumerate(memory):
        ret_mem, score, margin = retrieve_with_conf(mem["visual_q"])
        e_out = generate_answer(edit_prompt(ret_mem, mem["visual_q"]))
        r_out = generate_answer(edit_prompt(ret_mem, mem["rephrase_q"]))
        p_out = generate_answer(edit_prompt(ret_mem, mem["comp_q"]))
        l_out = generate_answer(simple_prompt(mem["loc_q"]))
        edit_ok.append(check_answer(e_out, mem["visual_a"]))
        rephrase_ok.append(check_answer(r_out, mem["visual_a"]))
        port_ok.append(check_answer(p_out, mem["comp_a"]))
        loc_ok.append(check_answer(l_out, mem["loc_a"]))
        if (i + 1) % 100 == 0:
            progress(i, len(memory), t0, "always_accept")
            if device == "cuda":
                torch.cuda.empty_cache()
    print(f"\n  [always_accept] edit={mean(edit_ok):.4f}  rephrase={mean(rephrase_ok):.4f}"
          f"  port={mean(port_ok):.4f}  loc={mean(loc_ok):.4f}")
    results["always_accept"] = {
        "edit_acc": round(mean(edit_ok), 4),
        "rephrase_acc": round(mean(rephrase_ok), 4),
        "portability_acc": round(mean(port_ok), 4),
        "locality_acc": round(mean(loc_ok), 4),
        "rejections": 0,
    }

    ablation = {}
    for tau in THRESHOLDS:
        edit_ok, rephrase_ok, port_ok, loc_ok = [], [], [], []
        rejections = 0
        t0 = time.time()
        for i, mem in enumerate(memory):
            ret_mem, score, margin = retrieve_with_conf(mem["visual_q"])
            if score < tau or margin < 0.05:
                e_out = generate_answer(simple_prompt(mem["visual_q"]))
                r_out = generate_answer(simple_prompt(mem["rephrase_q"]))
                p_out = generate_answer(simple_prompt(mem["comp_q"]))
                rejections += 1
            else:
                e_out = generate_answer(edit_prompt(ret_mem, mem["visual_q"]))
                r_out = generate_answer(edit_prompt(ret_mem, mem["rephrase_q"]))
                p_out = generate_answer(edit_prompt(ret_mem, mem["comp_q"]))
            l_out = generate_answer(simple_prompt(mem["loc_q"]))
            edit_ok.append(check_answer(e_out, mem["visual_a"]))
            rephrase_ok.append(check_answer(r_out, mem["visual_a"]))
            port_ok.append(check_answer(p_out, mem["comp_a"]))
            loc_ok.append(check_answer(l_out, mem["loc_a"]))
            if (i + 1) % 100 == 0:
                progress(i, len(memory), t0, f"tau={tau}")
                if device == "cuda":
                    torch.cuda.empty_cache()
        label = f"tau{int(tau*10)}"
        print(f"\n  [tau={tau}] edit={mean(edit_ok):.4f}  rejections={rejections}")
        ablation[label] = {
            "tau": tau,
            "edit_acc": round(mean(edit_ok), 4),
            "rephrase_acc": round(mean(rephrase_ok), 4),
            "portability_acc": round(mean(port_ok), 4),
            "locality_acc": round(mean(loc_ok), 4),
            "rejections": rejections,
        }

    best_tau_key = max(ablation, key=lambda k: ablation[k]["edit_acc"])
    best = ablation[best_tau_key]

    ck = {
        "experiment": "exp4_confidence_threshold",
        "dataset": DATA_PATH,
        "model": MODEL_NAME,
        "embedder": "all-MiniLM-L6-v2",
        "timestamp": datetime.now().isoformat(),
        "results_summary": {
            "always_accept_edit_acc": results["always_accept"]["edit_acc"],
            "best_threshold_edit_acc": best["edit_acc"],
            "best_tau": best["tau"],
            "delta_edit_acc": round(best["edit_acc"] - results["always_accept"]["edit_acc"], 4),
        },
        "results_detail": results,
        "ablation": ablation,
    }
    out_path = os.path.join(OUT_DIR, "adv2k_exp4_checkpoint.json")
    with open(out_path, "w") as f:
        json.dump(ck, f, indent=2)
    print(f"  Saved -> {out_path}")
    return results, ablation


# -
# MAIN
# -
if __name__ == "__main__":
    t_total = time.time()

    force_rerun = os.environ.get("ADVK2_FORCE_RERUN", "0") == "1"
    print(f"Resume mode: {'OFF (force rerun)' if force_rerun else 'ON (skip existing checkpoints)'}")

    print("\nBuilding memory bank from adversarial_2k.json ...")
    memory, vis_embs, txt_embs = build_memory(data)
    print(f"Memory bank ready: {len(memory)} entries\n")

    # Category distribution check
    cats = {}
    for s in data:
        c = get_category(s)
        cats[c] = cats.get(c, 0) + 1
    print("Category distribution:", cats)

    exp1_path = os.path.join(OUT_DIR, "adv2k_exp1_checkpoint.json")
    exp2_path = os.path.join(OUT_DIR, "adv2k_exp2_checkpoint.json")
    exp3_path = os.path.join(OUT_DIR, "adv2k_exp3_checkpoint.json")
    exp4_path = os.path.join(OUT_DIR, "adv2k_exp4_checkpoint.json")

    exp1_ck = None if force_rerun else load_json_if_exists(exp1_path)
    exp2_ck = None if force_rerun else load_json_if_exists(exp2_path)
    exp3_ck = None if force_rerun else load_json_if_exists(exp3_path)
    exp4_ck = None if force_rerun else load_json_if_exists(exp4_path)

    if exp1_ck:
        print(f"\n[skip] Found existing {exp1_path}")
        exp1_res = exp1_ck["results_detail"]
    else:
        exp1_res = run_exp1(memory, vis_embs, txt_embs)

    if exp2_ck:
        print(f"\n[skip] Found existing {exp2_path}")
        exp2_res = exp2_ck["results_detail"]
    else:
        exp2_res = run_exp2(memory, vis_embs, txt_embs)

    if exp3_ck:
        print(f"\n[skip] Found existing {exp3_path}")
        exp3_res = exp3_ck["results_detail"]
    else:
        exp3_res = run_exp3(memory, vis_embs, txt_embs)

    if exp4_ck:
        print(f"\n[skip] Found existing {exp4_path}")
        exp4_res = exp4_ck.get("results_detail", {})
        ablation = exp4_ck.get("ablation", {})
    else:
        exp4_res, ablation = run_exp4(memory, vis_embs, txt_embs)

    # - Summary table -
    print("\n" + "=" * 65)
    print("  SUMMARY -- adversarial_2k.json (Phi-2, text-only)")
    print("=" * 65)
    print(f"  Exp1  Baseline edit acc   : {exp1_res['baseline']['edit_acc']:.4f}")
    print(f"  Exp1  Adaptive gate acc   : {exp1_res['adaptive']['edit_acc']:.4f}")
    print(f"  Exp2  Hard-max edit acc   : {exp2_res['hard_max']['edit_acc']:.4f}")
    print(f"  Exp2  Soft top-k edit acc : {exp2_res['soft_topk']['edit_acc']:.4f}")
    print(f"  Exp3  No-connector acc    : {exp3_res['no_connector']['edit_acc']:.4f}")
    print(f"  Exp3  Gated connector acc : {exp3_res['gated_connector']['edit_acc']:.4f}")
    print(f"  Exp4  Always-accept acc   : {exp4_res['always_accept']['edit_acc']:.4f}")
    best_tau = max(ablation, key=lambda k: ablation[k]["edit_acc"])
    print(f"  Exp4  Best threshold acc  : {ablation[best_tau]['edit_acc']:.4f}  (tau={ablation[best_tau]['tau']})")
    print(f"\n  Total time: {(time.time()-t_total)/60:.1f} min")
    print("=" * 65)

    # - Master summary checkpoint -
    summary = {
        "dataset": DATA_PATH,
        "model": MODEL_NAME,
        "n_samples": len(data),
        "category_distribution": cats,
        "timestamp": datetime.now().isoformat(),
        "exp1": exp1_res,
        "exp2": exp2_res,
        "exp3": exp3_res,
        "exp4_baseline": exp4_res,
        "exp4_ablation": ablation,
    }
    out_path = os.path.join(OUT_DIR, "adv2k_summary_checkpoint.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Master summary -> {out_path}")

