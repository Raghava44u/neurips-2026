"""
All 4 Experiments with LLaVA-1.5-7B on train2017 dataset
=========================================================
Uses the actual LLaVA multimodal model (llava-hf/llava-1.5-7b-hf) on GPU.
Processes real train2017 COCO images through the vision encoder.
Results saved to new-checkpoint/results/llava/
"""

import json, random, os, sys, time, torch, gc
from datetime import datetime
from statistics import mean
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
SEED = 2026
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_PATH = "new-checkpoint/train2017_adversarial_2k.json"
IMAGE_ROOT = "datasets"
RESULTS_DIR = "new-checkpoint/results/llava"
PLOTS_DIR = "new-checkpoint/results/llava/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load Dataset ────────────────────────────────────────────────────
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Loaded {len(data)} train2017-based adversarial samples")

# ── Load LLaVA Model ───────────────────────────────────────────────
print(f"\nLoading LLaVA model: {LLAVA_MODEL_ID}")
print(f"Device: {device}")
t_load = time.time()

from transformers import LlavaForConditionalGeneration
from transformers import AutoTokenizer, CLIPImageProcessor

llava_model = LlavaForConditionalGeneration.from_pretrained(
    LLAVA_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
# Load tokenizer and image processor separately to avoid network issues
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# Model constants for manual image-token expansion
IMAGE_TOKEN_ID = llava_model.config.image_token_index  # 32000
vc = llava_model.config.vision_config
NUM_IMAGE_TOKENS = (vc.image_size // vc.patch_size) ** 2  # 576

llava_model.eval()
print(f"LLaVA loaded in {time.time() - t_load:.1f}s")

# Load SentenceTransformer for retrieval (same across all experiments)
print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(f"All models loaded on {device}\n")


# ═══════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════

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
            "image": s["image"],
        }
        entries.append(entry)
        vis_embs.append(embedder.encode(entry["visual_q"], convert_to_tensor=True))
        txt_embs.append(embedder.encode(entry["text_q"], convert_to_tensor=True))
    return entries, vis_embs, txt_embs


def load_image(image_path):
    """Load a train2017 image, return PIL Image or None."""
    full_path = os.path.join(IMAGE_ROOT, image_path)
    if os.path.exists(full_path):
        return Image.open(full_path).convert("RGB")
    return None


def generate_answer_with_image(prompt_text, image_path, max_tokens=30):
    """Generate answer using LLaVA with image + text (manual token expansion)."""
    img = load_image(image_path)
    if img is None:
        return generate_answer_text_only(prompt_text)

    # Process image through CLIP
    pixel_values = image_processor(images=img, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(llava_model.device, torch.float16)

    # Build input_ids manually: [BOS] USER: <image>*576 \n{prompt} ASSISTANT:
    before = tokenizer("USER: ", return_tensors="pt", add_special_tokens=True)["input_ids"]  # includes BOS
    after = tokenizer(f"\n{prompt_text} ASSISTANT:", return_tensors="pt", add_special_tokens=False)["input_ids"]
    img_ids = torch.full((1, NUM_IMAGE_TOKENS), IMAGE_TOKEN_ID, dtype=torch.long)
    input_ids = torch.cat([before, img_ids, after], dim=1).to(llava_model.device)

    with torch.no_grad():
        out = llava_model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=max_tokens, do_sample=False)
    generated = out[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    return text.split("\n")[0].strip()


def generate_answer_text_only(prompt_text, max_tokens=30):
    """Generate answer using LLaVA text-only (no image)."""
    text_prompt = f"USER: {prompt_text} ASSISTANT:"
    input_ids = tokenizer(text_prompt, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(llava_model.device)

    with torch.no_grad():
        out = llava_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens, do_sample=False)
    generated = out[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
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


# Precompute memory embeddings once
print("Building memory embeddings...")
t0 = time.time()
memory, vis_embs, txt_embs = build_memory_split(data)
print(f"Memory built in {time.time()-t0:.1f}s ({len(memory)} entries)\n")

metrics_keys = ["edit_acc", "rephrase_acc", "locality_acc",
                "portability_acc", "retrieval_score"]


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  EXPERIMENT 1: ADAPTIVE MODALITY GATING                         ║
# ╚═══════════════════════════════════════════════════════════════════╝

def run_experiment_1():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1 (LLaVA): BASELINE vs ADAPTIVE MODALITY GATING")
    print(f"  Dataset: train2017 adversarial ({len(data)} samples)")
    print("=" * 70)

    def retrieve_baseline(query, vis_embs, txt_embs):
        q_emb = embedder.encode(query, convert_to_tensor=True)
        vis_scores = np.array([util.cos_sim(q_emb, v).item() for v in vis_embs])
        txt_scores = np.array([util.cos_sim(q_emb, t).item() for t in txt_embs])
        final_scores = 0.1 * txt_scores + 0.9 * vis_scores
        best_idx = int(np.argmax(final_scores))
        vis_best = int(np.argmax(vis_scores))
        txt_best = int(np.argmax(txt_scores))
        return {"entry": memory[best_idx], "score": float(final_scores[best_idx]),
                "idx": best_idx, "vis_best": vis_best, "txt_best": txt_best,
                "modalities_agree": vis_best == txt_best, "alpha_used": 0.9}

    def retrieve_adaptive(query, vis_embs, txt_embs):
        q_emb = embedder.encode(query, convert_to_tensor=True)
        vis_scores = np.array([util.cos_sim(q_emb, v).item() for v in vis_embs])
        txt_scores = np.array([util.cos_sim(q_emb, t).item() for t in txt_embs])
        vis_best = int(np.argmax(vis_scores))
        txt_best = int(np.argmax(txt_scores))
        agree = (vis_best == txt_best)
        alpha = 0.9 if agree else 0.5
        final_scores = (1.0 - alpha) * txt_scores + alpha * vis_scores
        best_idx = int(np.argmax(final_scores))
        return {"entry": memory[best_idx], "score": float(final_scores[best_idx]),
                "idx": best_idx, "vis_best": vis_best, "txt_best": txt_best,
                "modalities_agree": agree, "alpha_used": alpha}

    results = {}
    for method_name, retrieve_fn in [("baseline", retrieve_baseline),
                                      ("adaptive", retrieve_adaptive)]:
        print(f"\n  ── Running: {method_name.upper()} ──")
        counters = {k: [] for k in metrics_keys}
        per_category = {}
        conflicts_detected = 0
        conflict_fixed = 0
        t0 = time.time()

        for i, mem in enumerate(memory):
            cat = get_category(data[i])
            if cat not in per_category:
                per_category[cat] = {k: [] for k in metrics_keys}

            ret = retrieve_fn(mem["visual_q"], vis_embs, txt_embs)
            entry = ret["entry"]
            counters["retrieval_score"].append(ret["score"])
            per_category[cat]["retrieval_score"].append(ret["score"])
            if not ret["modalities_agree"]:
                conflicts_detected += 1

            # Edit (with image)
            e_out = generate_answer_with_image(
                build_edit_prompt(entry, mem["visual_q"]), mem["image"])
            e_ok = check_answer(e_out, mem["visual_a"])
            counters["edit_acc"].append(int(e_ok))
            per_category[cat]["edit_acc"].append(int(e_ok))

            # Rephrase (with image)
            r_out = generate_answer_with_image(
                build_edit_prompt(entry, mem["rephrase_q"]), mem["image"])
            r_ok = check_answer(r_out, mem["visual_a"])
            counters["rephrase_acc"].append(int(r_ok))
            per_category[cat]["rephrase_acc"].append(int(r_ok))

            # Locality (text-only)
            l_out = generate_answer_text_only(build_simple_prompt(mem["loc_q"]))
            l_ok = check_answer(l_out, mem["loc_a"])
            counters["locality_acc"].append(int(l_ok))
            per_category[cat]["locality_acc"].append(int(l_ok))

            # Portability (with image)
            p_out = generate_answer_with_image(
                build_edit_prompt(entry, mem["comp_q"]), mem["image"])
            p_ok = check_answer(p_out, mem["comp_a"])
            counters["portability_acc"].append(int(p_ok))
            per_category[cat]["portability_acc"].append(int(p_ok))

            if method_name == "adaptive" and not ret["modalities_agree"] and e_ok:
                conflict_fixed += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"    [{i+1:>4d}/{len(data)}] {elapsed:>6.0f}s | "
                      f"edit={mean(counters['edit_acc']):.3f} "
                      f"reph={mean(counters['rephrase_acc']):.3f} "
                      f"port={mean(counters['portability_acc']):.3f}")

        elapsed = time.time() - t0
        overall = {k: round(mean(v), 4) for k, v in counters.items()}
        overall["conflicts_detected"] = conflicts_detected
        overall["conflict_fixed"] = conflict_fixed if method_name == "adaptive" else "N/A"
        overall["eval_time_seconds"] = round(elapsed, 2)

        cat_summary = {}
        for cat, cnt in per_category.items():
            cat_summary[cat] = {k: round(mean(v), 4) for k, v in cnt.items() if v}
            cat_summary[cat]["count"] = len(cnt["edit_acc"])

        results[method_name] = {"overall": overall, "per_category": cat_summary}

        print(f"\n  {method_name.upper()} Results:")
        for k in metrics_keys:
            v = overall[k]
            if k.endswith("_acc"):
                print(f"    {k:<20s} {v*100:>6.1f}%")
            else:
                print(f"    {k:<20s} {v:>6.4f}")

    # Deltas
    b, a = results["baseline"]["overall"], results["adaptive"]["overall"]
    deltas = {k: round(a[k] - b[k], 4) for k in metrics_keys}
    print(f"\n  DELTA (Adaptive - Baseline):")
    for k in metrics_keys:
        d = deltas[k]
        if k.endswith("_acc"):
            print(f"    {k:<20s} {d*100:>+6.1f}pp")

    # Save
    save_obj = {
        "experiment": "Exp1: Adaptive Modality Gating (LLaVA)",
        "timestamp": datetime.now().isoformat(),
        "dataset": f"train2017_adversarial_2k ({len(data)} samples)",
        "model": LLAVA_MODEL_ID, "device": device,
        "baseline": results["baseline"], "adaptive": results["adaptive"],
        "deltas": deltas,
    }
    with open(f"{RESULTS_DIR}/exp1_adaptive_gating.json", "w") as f:
        json.dump(save_obj, f, indent=2)
    print(f"  Saved: {RESULTS_DIR}/exp1_adaptive_gating.json")

    ckpt = {
        "experiment": "exp1_adaptive_gating_llava",
        "model": LLAVA_MODEL_ID,
        "timestamp": datetime.now().isoformat(),
        "results_summary": {
            "baseline_edit_acc": b["edit_acc"],
            "adaptive_edit_acc": a["edit_acc"],
            "delta": deltas["edit_acc"],
        },
    }
    with open("new-checkpoint/exp1_llava_checkpoint.json", "w") as f:
        json.dump(ckpt, f, indent=2)

    return results, deltas


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  EXPERIMENT 2: SOFT TOP-K RETRIEVAL                             ║
# ╚═══════════════════════════════════════════════════════════════════╝

def run_experiment_2():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2 (LLaVA): HARD MAX vs SOFT TOP-K RETRIEVAL")
    print(f"  Dataset: train2017 adversarial ({len(data)} samples)")
    print("=" * 70)

    TOPK_CONFIG = {"k": 3, "temperature": 0.1, "ambiguity_threshold": 0.7}

    def retrieve_hard_max(query):
        q_emb = embedder.encode(query, convert_to_tensor=True)
        scores = np.array([max(util.cos_sim(q_emb, v).item(),
                               util.cos_sim(q_emb, t).item())
                           for v, t in zip(vis_embs, txt_embs)])
        best_idx = int(np.argmax(scores))
        sorted_idx = np.argsort(scores)[::-1]
        margin = float(scores[sorted_idx[0]] - scores[sorted_idx[1]]) if len(scores) > 1 else 1.0
        return {"entry": memory[best_idx], "score": float(scores[best_idx]),
                "idx": best_idx, "margin": margin, "is_ambiguous": False, "top1_weight": 1.0}

    def retrieve_soft_topk(query, k=3, temperature=0.1, ambiguity_threshold=0.7):
        q_emb = embedder.encode(query, convert_to_tensor=True)
        scores = np.array([max(util.cos_sim(q_emb, v).item(),
                               util.cos_sim(q_emb, t).item())
                           for v, t in zip(vis_embs, txt_embs)])
        topk_idx = np.argsort(scores)[::-1][:k]
        topk_scores = scores[topk_idx]
        exp_scores = np.exp((topk_scores - topk_scores.max()) / temperature)
        weights = exp_scores / exp_scores.sum()
        top1_weight = float(weights[0])
        is_ambiguous = top1_weight < ambiguity_threshold
        margin = float(topk_scores[0] - topk_scores[1]) if k > 1 else 1.0
        best_idx = int(topk_idx[0])
        return {"entry": memory[best_idx], "score": float(topk_scores[0]),
                "idx": best_idx, "margin": margin, "is_ambiguous": is_ambiguous,
                "top1_weight": top1_weight}

    results = {}
    for method_name, retrieve_fn in [("hard_max", lambda q: retrieve_hard_max(q)),
                                      ("soft_topk", lambda q: retrieve_soft_topk(q, **TOPK_CONFIG))]:
        print(f"\n  ── Running: {method_name.upper()} ──")
        counters = {k: [] for k in metrics_keys}
        per_category = {}
        ambiguous_count = 0
        t0 = time.time()

        for i, mem in enumerate(memory):
            cat = get_category(data[i])
            if cat not in per_category:
                per_category[cat] = {k: [] for k in metrics_keys}

            ret = retrieve_fn(mem["visual_q"])
            entry = ret["entry"]
            counters["retrieval_score"].append(ret["score"])
            per_category[cat]["retrieval_score"].append(ret["score"])
            if ret.get("is_ambiguous", False):
                ambiguous_count += 1

            e_out = generate_answer_with_image(
                build_edit_prompt(entry, mem["visual_q"]), mem["image"])
            e_ok = check_answer(e_out, mem["visual_a"])
            counters["edit_acc"].append(int(e_ok))
            per_category[cat]["edit_acc"].append(int(e_ok))

            r_out = generate_answer_with_image(
                build_edit_prompt(entry, mem["rephrase_q"]), mem["image"])
            r_ok = check_answer(r_out, mem["visual_a"])
            counters["rephrase_acc"].append(int(r_ok))
            per_category[cat]["rephrase_acc"].append(int(r_ok))

            l_out = generate_answer_text_only(build_simple_prompt(mem["loc_q"]))
            l_ok = check_answer(l_out, mem["loc_a"])
            counters["locality_acc"].append(int(l_ok))
            per_category[cat]["locality_acc"].append(int(l_ok))

            p_out = generate_answer_with_image(
                build_edit_prompt(entry, mem["comp_q"]), mem["image"])
            p_ok = check_answer(p_out, mem["comp_a"])
            counters["portability_acc"].append(int(p_ok))
            per_category[cat]["portability_acc"].append(int(p_ok))

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"    [{i+1:>4d}/{len(data)}] {elapsed:>6.0f}s | "
                      f"edit={mean(counters['edit_acc']):.3f} "
                      f"reph={mean(counters['rephrase_acc']):.3f}")

        elapsed = time.time() - t0
        overall = {k: round(mean(v), 4) for k, v in counters.items()}
        overall["ambiguous_count"] = ambiguous_count
        overall["eval_time_seconds"] = round(elapsed, 2)

        cat_summary = {}
        for cat, cnt in per_category.items():
            cat_summary[cat] = {k: round(mean(v), 4) for k, v in cnt.items() if v}
            cat_summary[cat]["count"] = len(cnt["edit_acc"])

        results[method_name] = {"overall": overall, "per_category": cat_summary}

        print(f"\n  {method_name.upper()} Results:")
        for k in metrics_keys:
            v = overall[k]
            if k.endswith("_acc"):
                print(f"    {k:<20s} {v*100:>6.1f}%")

    b, a = results["hard_max"]["overall"], results["soft_topk"]["overall"]
    deltas = {k: round(a[k] - b[k], 4) for k in metrics_keys}
    print(f"\n  DELTA (Soft TopK - Hard Max):")
    for k in metrics_keys:
        if k.endswith("_acc"):
            print(f"    {k:<20s} {deltas[k]*100:>+6.1f}pp")

    save_obj = {
        "experiment": "Exp2: Soft Top-K Retrieval (LLaVA)",
        "timestamp": datetime.now().isoformat(),
        "dataset": f"train2017_adversarial_2k ({len(data)} samples)",
        "model": LLAVA_MODEL_ID, "device": device,
        "hyperparameters": TOPK_CONFIG,
        "hard_max": results["hard_max"], "soft_topk": results["soft_topk"],
        "deltas": deltas,
    }
    with open(f"{RESULTS_DIR}/exp2_soft_topk.json", "w") as f:
        json.dump(save_obj, f, indent=2)
    print(f"  Saved: {RESULTS_DIR}/exp2_soft_topk.json")

    ckpt = {
        "experiment": "exp2_soft_topk_llava",
        "model": LLAVA_MODEL_ID,
        "timestamp": datetime.now().isoformat(),
        "results_summary": {
            "hard_max_edit_acc": b["edit_acc"],
            "soft_topk_edit_acc": a["edit_acc"],
            "delta": deltas["edit_acc"],
        },
    }
    with open("new-checkpoint/exp2_llava_checkpoint.json", "w") as f:
        json.dump(ckpt, f, indent=2)

    return results, deltas


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  EXPERIMENT 3: CONSISTENCY-CHECKED CONNECTOR                    ║
# ╚═══════════════════════════════════════════════════════════════════╝

def run_experiment_3():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3 (LLaVA): CONSISTENCY-CHECKED CONNECTOR")
    print(f"  Dataset: train2017 adversarial ({len(data)} samples)")
    print("=" * 70)

    CONSISTENCY_THRESHOLD = 0.5

    def retrieve_best(query):
        q_emb = embedder.encode(query, convert_to_tensor=True)
        vis_scores = np.array([util.cos_sim(q_emb, v).item() for v in vis_embs])
        txt_scores = np.array([util.cos_sim(q_emb, t).item() for t in txt_embs])
        vis_best = int(np.argmax(vis_scores))
        txt_best = int(np.argmax(txt_scores))
        return {"vis_entry": memory[vis_best], "txt_entry": memory[txt_best],
                "vis_score": float(vis_scores[vis_best]),
                "txt_score": float(txt_scores[txt_best]),
                "vis_idx": vis_best, "txt_idx": txt_best}

    def build_both_prompt(vis_entry, txt_entry, question):
        return (f"Use the following facts to answer accurately.\n\n"
                f"Visual fact: {vis_entry['visual_q']} → {vis_entry['visual_a']}\n"
                f"Text fact: {txt_entry['text_q']} → {txt_entry['text_a']}\n\n"
                f"Question: {question}\nAnswer:")

    def build_single_prompt(entry, question, modality="visual"):
        if modality == "visual":
            fact = f"{entry['visual_q']} → {entry['visual_a']}"
        else:
            fact = f"{entry['text_q']} → {entry['text_a']}"
        return (f"Use the following fact to answer accurately.\n\n"
                f"Fact: {fact}\n\nQuestion: {question}\nAnswer:")

    methods = ["standard", "no_connector", "gated"]
    results = {}

    for method_name in methods:
        print(f"\n  ── Running: {method_name.upper()} ──")
        counters = {k: [] for k in metrics_keys}
        per_category = {}
        bypass_count = 0
        t0 = time.time()

        for i, mem in enumerate(memory):
            cat = get_category(data[i])
            if cat not in per_category:
                per_category[cat] = {k: [] for k in metrics_keys}

            ret = retrieve_best(mem["visual_q"])
            score = max(ret["vis_score"], ret["txt_score"])
            counters["retrieval_score"].append(score)
            per_category[cat]["retrieval_score"].append(score)

            img_path = mem["image"]

            # Edit
            if method_name == "standard":
                prompt = build_both_prompt(ret["vis_entry"], ret["txt_entry"], mem["visual_q"])
                e_out = generate_answer_with_image(prompt, img_path)
            elif method_name == "no_connector":
                if ret["vis_score"] >= ret["txt_score"]:
                    prompt = build_single_prompt(ret["vis_entry"], mem["visual_q"], "visual")
                else:
                    prompt = build_single_prompt(ret["txt_entry"], mem["visual_q"], "text")
                e_out = generate_answer_with_image(prompt, img_path)
                bypass_count += 1
            else:  # gated
                vis_a_emb = embedder.encode(ret["vis_entry"]["visual_a"], convert_to_tensor=True)
                txt_a_emb = embedder.encode(ret["txt_entry"]["text_a"], convert_to_tensor=True)
                consistency = util.cos_sim(vis_a_emb, txt_a_emb).item()
                if consistency >= CONSISTENCY_THRESHOLD:
                    prompt = build_both_prompt(ret["vis_entry"], ret["txt_entry"], mem["visual_q"])
                else:
                    bypass_count += 1
                    if ret["vis_score"] >= ret["txt_score"]:
                        prompt = build_single_prompt(ret["vis_entry"], mem["visual_q"], "visual")
                    else:
                        prompt = build_single_prompt(ret["txt_entry"], mem["visual_q"], "text")
                e_out = generate_answer_with_image(prompt, img_path)

            e_ok = check_answer(e_out, mem["visual_a"])
            counters["edit_acc"].append(int(e_ok))
            per_category[cat]["edit_acc"].append(int(e_ok))

            # Rephrase (with image)
            r_prompt = build_both_prompt(ret["vis_entry"], ret["txt_entry"], mem["rephrase_q"])
            r_out = generate_answer_with_image(r_prompt, img_path)
            r_ok = check_answer(r_out, mem["visual_a"])
            counters["rephrase_acc"].append(int(r_ok))
            per_category[cat]["rephrase_acc"].append(int(r_ok))

            # Locality (text-only)
            l_out = generate_answer_text_only(build_simple_prompt(mem["loc_q"]))
            l_ok = check_answer(l_out, mem["loc_a"])
            counters["locality_acc"].append(int(l_ok))
            per_category[cat]["locality_acc"].append(int(l_ok))

            # Portability (with image)
            p_prompt = build_both_prompt(ret["vis_entry"], ret["txt_entry"], mem["comp_q"])
            p_out = generate_answer_with_image(p_prompt, img_path)
            p_ok = check_answer(p_out, mem["comp_a"])
            counters["portability_acc"].append(int(p_ok))
            per_category[cat]["portability_acc"].append(int(p_ok))

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"    [{i+1:>4d}/{len(data)}] {elapsed:>6.0f}s | "
                      f"edit={mean(counters['edit_acc']):.3f}")

        elapsed = time.time() - t0
        overall = {k: round(mean(v), 4) for k, v in counters.items()}
        overall["bypass_count"] = bypass_count
        overall["eval_time_seconds"] = round(elapsed, 2)

        cat_summary = {}
        for cat, cnt in per_category.items():
            cat_summary[cat] = {k: round(mean(v), 4) for k, v in cnt.items() if v}
            cat_summary[cat]["count"] = len(cnt["edit_acc"])

        results[method_name] = {"overall": overall, "per_category": cat_summary}

        print(f"\n  {method_name.upper()} Results:")
        for k in metrics_keys:
            v = overall[k]
            if k.endswith("_acc"):
                print(f"    {k:<20s} {v*100:>6.1f}%")

    s_ovr = results["standard"]["overall"]
    n_ovr = results["no_connector"]["overall"]
    g_ovr = results["gated"]["overall"]
    deltas = {k: round(g_ovr[k] - s_ovr[k], 4) for k in metrics_keys}

    print(f"\n  COMPARISON:")
    print(f"  {'Metric':<20s} {'Standard':>10s} {'No-Conn':>10s} {'Gated':>10s}")
    for k in metrics_keys:
        if k.endswith("_acc"):
            print(f"  {k:<20s} {s_ovr[k]*100:>9.1f}% {n_ovr[k]*100:>9.1f}% {g_ovr[k]*100:>9.1f}%")

    save_obj = {
        "experiment": "Exp3: Consistency Connector (LLaVA)",
        "timestamp": datetime.now().isoformat(),
        "dataset": f"train2017_adversarial_2k ({len(data)} samples)",
        "model": LLAVA_MODEL_ID, "device": device,
        "standard": results["standard"], "no_connector": results["no_connector"],
        "gated": results["gated"], "deltas": deltas,
    }
    with open(f"{RESULTS_DIR}/exp3_consistency_connector.json", "w") as f:
        json.dump(save_obj, f, indent=2)
    print(f"  Saved: {RESULTS_DIR}/exp3_consistency_connector.json")

    ckpt = {
        "experiment": "exp3_consistency_connector_llava",
        "model": LLAVA_MODEL_ID,
        "timestamp": datetime.now().isoformat(),
        "results_summary": {
            "standard_edit_acc": s_ovr["edit_acc"],
            "no_connector_edit_acc": n_ovr["edit_acc"],
            "gated_edit_acc": g_ovr["edit_acc"],
        },
    }
    with open("new-checkpoint/exp3_llava_checkpoint.json", "w") as f:
        json.dump(ckpt, f, indent=2)

    return results, deltas


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  EXPERIMENT 4: RETRIEVAL CONFIDENCE THRESHOLD                   ║
# ╚═══════════════════════════════════════════════════════════════════╝

def run_experiment_4():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4 (LLaVA): RETRIEVAL CONFIDENCE THRESHOLD")
    print(f"  Dataset: train2017 adversarial ({len(data)} samples)")
    print("=" * 70)

    threshold_configs = [
        ("always_accept", 0.0, 0.0),
        ("threshold_0.5", 0.5, 0.03),
        ("threshold_0.6", 0.6, 0.05),
        ("threshold_0.7", 0.7, 0.08),
    ]

    def retrieve_with_confidence(query):
        q_emb = embedder.encode(query, convert_to_tensor=True)
        scores = np.array([max(util.cos_sim(q_emb, v).item(),
                               util.cos_sim(q_emb, t).item())
                           for v, t in zip(vis_embs, txt_embs)])
        sorted_idx = np.argsort(scores)[::-1]
        best_idx = int(sorted_idx[0])
        margin = float(scores[sorted_idx[0]] - scores[sorted_idx[1]]) if len(scores) > 1 else 1.0
        return {"entry": memory[best_idx], "score": float(scores[best_idx]),
                "idx": best_idx, "margin": margin}

    results = {}
    for config_name, min_conf, min_marg in threshold_configs:
        print(f"\n  ── Running: {config_name.upper()} (conf={min_conf}, marg={min_marg}) ──")
        counters = {k: [] for k in metrics_keys}
        per_category = {}
        rejection_count = 0
        t0 = time.time()

        for i, mem in enumerate(memory):
            cat = get_category(data[i])
            if cat not in per_category:
                per_category[cat] = {k: [] for k in metrics_keys}

            ret = retrieve_with_confidence(mem["visual_q"])
            counters["retrieval_score"].append(ret["score"])
            per_category[cat]["retrieval_score"].append(ret["score"])

            rejected = (min_conf > 0 and
                       (ret["score"] < min_conf or ret["margin"] < min_marg))

            img_path = mem["image"]

            # Edit
            if rejected:
                rejection_count += 1
                e_out = generate_answer_text_only(build_simple_prompt(mem["visual_q"]))
            else:
                e_out = generate_answer_with_image(
                    build_edit_prompt(ret["entry"], mem["visual_q"]), img_path)
            e_ok = check_answer(e_out, mem["visual_a"])
            counters["edit_acc"].append(int(e_ok))
            per_category[cat]["edit_acc"].append(int(e_ok))

            # Rephrase
            if rejected:
                r_out = generate_answer_text_only(build_simple_prompt(mem["rephrase_q"]))
            else:
                r_out = generate_answer_with_image(
                    build_edit_prompt(ret["entry"], mem["rephrase_q"]), img_path)
            r_ok = check_answer(r_out, mem["visual_a"])
            counters["rephrase_acc"].append(int(r_ok))
            per_category[cat]["rephrase_acc"].append(int(r_ok))

            # Locality (always text-only)
            l_out = generate_answer_text_only(build_simple_prompt(mem["loc_q"]))
            l_ok = check_answer(l_out, mem["loc_a"])
            counters["locality_acc"].append(int(l_ok))
            per_category[cat]["locality_acc"].append(int(l_ok))

            # Portability
            if rejected:
                p_out = generate_answer_text_only(build_simple_prompt(mem["comp_q"]))
            else:
                p_out = generate_answer_with_image(
                    build_edit_prompt(ret["entry"], mem["comp_q"]), img_path)
            p_ok = check_answer(p_out, mem["comp_a"])
            counters["portability_acc"].append(int(p_ok))
            per_category[cat]["portability_acc"].append(int(p_ok))

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"    [{i+1:>4d}/{len(data)}] {elapsed:>6.0f}s | "
                      f"edit={mean(counters['edit_acc']):.3f} "
                      f"reject={rejection_count}")

        elapsed = time.time() - t0
        overall = {k: round(mean(v), 4) for k, v in counters.items()}
        overall["rejection_count"] = rejection_count
        overall["rejection_rate"] = round(rejection_count / len(data), 4)
        overall["eval_time_seconds"] = round(elapsed, 2)

        cat_summary = {}
        for cat, cnt in per_category.items():
            cat_summary[cat] = {k: round(mean(v), 4) for k, v in cnt.items() if v}
            cat_summary[cat]["count"] = len(cnt["edit_acc"])

        results[config_name] = {"overall": overall, "per_category": cat_summary}

        print(f"\n  {config_name.upper()} Results:")
        for k in metrics_keys:
            v = overall[k]
            if k.endswith("_acc"):
                print(f"    {k:<20s} {v*100:>6.1f}%")
        print(f"    {'rejections':<20s} {rejection_count} ({overall['rejection_rate']*100:.1f}%)")

    b = results["always_accept"]["overall"]
    a = results["threshold_0.6"]["overall"]
    deltas = {k: round(a[k] - b[k], 4) for k in metrics_keys}

    print(f"\n  PRIMARY DELTA (Threshold 0.6 - Always Accept):")
    for k in metrics_keys:
        if k.endswith("_acc"):
            print(f"    {k:<20s} {deltas[k]*100:>+6.1f}pp")

    save_obj = {
        "experiment": "Exp4: Confidence Threshold (LLaVA)",
        "timestamp": datetime.now().isoformat(),
        "dataset": f"train2017_adversarial_2k ({len(data)} samples)",
        "model": LLAVA_MODEL_ID, "device": device,
        "threshold_configs": {n: {"min_confidence": c, "min_margin": m}
                             for n, c, m in threshold_configs},
    }
    for cn, _, _ in threshold_configs:
        save_obj[cn] = results[cn]
    save_obj["deltas"] = deltas

    with open(f"{RESULTS_DIR}/exp4_confidence_threshold.json", "w") as f:
        json.dump(save_obj, f, indent=2)
    print(f"  Saved: {RESULTS_DIR}/exp4_confidence_threshold.json")

    ckpt = {
        "experiment": "exp4_confidence_threshold_llava",
        "model": LLAVA_MODEL_ID,
        "timestamp": datetime.now().isoformat(),
        "results_summary": {
            "always_accept_edit_acc": b["edit_acc"],
            "threshold_0.6_edit_acc": a["edit_acc"],
            "delta": deltas["edit_acc"],
            "rejection_rate": a["rejection_rate"],
        },
        "ablation": {n: results[n]["overall"]["edit_acc"]
                     for n, _, _ in threshold_configs},
    }
    with open("new-checkpoint/exp4_llava_checkpoint.json", "w") as f:
        json.dump(ckpt, f, indent=2)

    return results, deltas


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  MAIN: RUN ALL EXPERIMENTS                                      ║
# ╚═══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 70)
    print("  LLaVA-1.5-7B EXPERIMENTS ON TRAIN2017 DATASET")
    print("  4 experiments × 2000 samples with multimodal processing")
    print("=" * 70)

    all_results = {}
    total_start = time.time()

    # Exp 1 & 2 already completed — load from saved results
    import glob
    for epath in sorted(glob.glob(os.path.join(RESULTS_DIR, "exp*.json"))):
        ename = os.path.basename(epath).replace(".json", "")
        with open(epath) as ef:
            saved = json.load(ef)
        key = "exp1" if "exp1" in ename else "exp2"
        all_results[key] = saved
        print(f"  [LOADED] {ename} from previous run")

    print("\n\n>>> STARTING EXPERIMENT 3...")
    r3, d3 = run_experiment_3()
    all_results["exp3"] = {"results": {k: v["overall"] for k, v in r3.items()}, "deltas": d3}
    gc.collect()
    torch.cuda.empty_cache()

    print("\n\n>>> STARTING EXPERIMENT 4...")
    r4, d4 = run_experiment_4()
    all_results["exp4"] = {"results": {k: v["overall"] for k, v in r4.items()}, "deltas": d4}

    total_elapsed = time.time() - total_start

    # ── Final Summary ──────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  FINAL SUMMARY — ALL 4 EXPERIMENTS (LLaVA-1.5-7B)")
    print("=" * 70)
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print(f"  Model: {LLAVA_MODEL_ID}")
    print(f"  Dataset: train2017 adversarial ({len(data)} samples)")
    print()

    summary_table = {
        "Exp1 (Adaptive Gating)": d1,
        "Exp2 (Soft Top-K)": d2,
        "Exp3 (Consistency Conn.)": d3,
        "Exp4 (Confidence Thr.)": d4,
    }

    print(f"  {'Experiment':<30s} {'Edit Δ':>10s} {'Reph Δ':>10s} {'Port Δ':>10s} {'Loc Δ':>10s}")
    print(f"  {'─'*72}")
    for exp_name, d in summary_table.items():
        print(f"  {exp_name:<30s} {d['edit_acc']*100:>+9.1f}pp "
              f"{d['rephrase_acc']*100:>+9.1f}pp "
              f"{d['portability_acc']*100:>+9.1f}pp "
              f"{d['locality_acc']*100:>+9.1f}pp")

    # Save master summary
    master = {
        "title": "LLaVA-1.5-7B Experiments on train2017 (2000 samples)",
        "timestamp": datetime.now().isoformat(),
        "model": LLAVA_MODEL_ID,
        "total_time_minutes": round(total_elapsed / 60, 1),
        "experiments": all_results,
    }
    with open(f"{RESULTS_DIR}/llava_master_summary.json", "w") as f:
        json.dump(master, f, indent=2)
    print(f"\n  Master summary: {RESULTS_DIR}/llava_master_summary.json")
    print("\n  DONE.")
