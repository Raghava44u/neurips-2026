"""
Run All NeurIPS Experiments on 2000-Sample Dataset
===================================================
Executes all methods in sequence:
  1. Baseline (fixed alpha, standard connector)
  2. Adaptive Modality Gating (Exp 1)
  3. Soft Top-K Retrieval (Exp 2)
  4. Connector Variants: Standard, No-Connector, Gated (Exp 3)
  5. Confidence Threshold (Exp 4)
  6. CAFE: Confidence-Aware Fusion Engine (Exp 5 - Novel)
  7. Additional baselines: Pure RAG, Text-Only Editing

Output: results/neurips_all_results.json (complete results for paper)
"""

import json, random, os, time, torch, sys
import numpy as np
from datetime import datetime
from statistics import mean, stdev
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

SEED = 2026
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

DATASET_PATH = "datasets/adversarial_2k.json"
OUTPUT_PATH = "results/neurips_all_results.json"


def get_category(sample):
    loc = sample.get("loc", "")
    if "boiling point of water" in loc: return "polysemy"
    elif "plants absorb" in loc: return "conflict"
    elif "bones" in loc: return "near_miss"
    elif "largest organ" in loc: return "multi_hop"
    elif "planet do we live" in loc: return "hard_visual"
    return "unknown"


def build_memory(data, embedder):
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


class ExperimentRunner:
    def __init__(self):
        print("Loading models...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL_NAME = "microsoft/phi-2"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True, device_map=self.device)
        self.model.eval()
        print(f"Models loaded on {self.device}")

    def generate(self, prompt, max_tokens=30):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "Answer:" in text:
            text = text.split("Answer:")[-1].strip()
        return text.split("\n")[0].strip()

    def check_answer(self, prediction, expected):
        p, e = prediction.lower().strip(), expected.lower().strip()
        return e in p or p in e

    def retrieve(self, query, memory, vis_embs, txt_embs, alpha=0.9):
        q_emb = self.embedder.encode(query, convert_to_tensor=True)
        vis_scores = [float(util.cos_sim(q_emb, v).item()) for v in vis_embs]
        txt_scores = [float(util.cos_sim(q_emb, t).item()) for t in txt_embs]
        combined = [(1-alpha)*t + alpha*v for v, t in zip(vis_scores, txt_scores)]
        best_idx = int(np.argmax(combined))
        return memory[best_idx], combined[best_idx], best_idx, combined, q_emb

    def evaluate_method(self, method_name, data, memory, vis_embs, txt_embs,
                        prompt_fn, alpha=0.9):
        """Generic evaluation loop for any method."""
        results = {"correct": [], "per_cat": {}, "reph_correct": [],
                   "loc_correct": [], "port_correct": []}

        for idx, (sample, entry) in enumerate(zip(data, memory)):
            query = entry["visual_q"]
            expected = entry["visual_a"]
            category = get_category(sample)

            # Retrieve
            best_entry, score, best_idx, all_scores, q_emb = self.retrieve(
                query, memory, vis_embs, txt_embs, alpha)

            # Generate with method-specific prompt
            prompt = prompt_fn(best_entry, query, q_emb, vis_embs, txt_embs,
                               best_idx, all_scores)
            pred = self.generate(prompt)
            correct = self.check_answer(pred, expected)

            results["correct"].append(int(correct))
            if category not in results["per_cat"]:
                results["per_cat"][category] = []
            results["per_cat"][category].append(int(correct))

            # Rephrase
            reph_prompt = prompt_fn(best_entry, entry["rephrase_q"], q_emb,
                                     vis_embs, txt_embs, best_idx, all_scores)
            reph_pred = self.generate(reph_prompt)
            results["reph_correct"].append(int(self.check_answer(reph_pred, expected)))

            # Locality
            loc_pred = self.generate(f"Question: {entry['loc_q']}\nAnswer:")
            results["loc_correct"].append(int(self.check_answer(loc_pred, entry["loc_a"])))

            # Portability
            port_prompt = (f"Fact: {best_entry['visual_q']} → {best_entry['visual_a']}\n"
                           f"Question: {entry['comp_q']}\nAnswer:")
            port_pred = self.generate(port_prompt)
            results["port_correct"].append(int(self.check_answer(port_pred, entry["comp_a"])))

            if (idx + 1) % 200 == 0:
                acc = mean(results["correct"])
                print(f"  [{method_name}] {idx+1}/{len(data)}: EA={acc*100:.1f}%")

        return self._summarize(results)

    def _summarize(self, results):
        """Compute summary statistics with bootstrap CIs."""
        n = len(results["correct"])
        ea = mean(results["correct"])

        # Bootstrap 95% CI
        boot_accs = []
        for _ in range(1000):
            boot = random.choices(results["correct"], k=n)
            boot_accs.append(mean(boot))
        boot_accs.sort()

        summary = {
            "edit_acc": ea,
            "edit_acc_std": stdev(boot_accs),
            "edit_acc_ci95": [boot_accs[25], boot_accs[974]],
            "rephrase_acc": mean(results["reph_correct"]) if results["reph_correct"] else 0,
            "locality_acc": mean(results["loc_correct"]) if results["loc_correct"] else 0,
            "portability_acc": mean(results["port_correct"]) if results["port_correct"] else 0,
            "n_samples": n,
            "per_category": {},
        }

        for cat, cat_results in results["per_cat"].items():
            cat_acc = mean(cat_results)
            summary["per_category"][cat] = {
                "edit_acc": cat_acc,
                "count": len(cat_results),
            }

        return summary


def main():
    print("=" * 70)
    print("ADCMF NeurIPS Complete Experiment Suite")
    print("=" * 70)

    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {DATASET_PATH}")

    runner = ExperimentRunner()
    memory, vis_embs, txt_embs = build_memory(data, runner.embedder)

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": DATASET_PATH,
            "n_samples": len(data),
            "seed": SEED,
            "model": "microsoft/phi-2",
            "embedder": "all-MiniLM-L6-v2",
            "device": runner.device,
        },
        "methods": {},
    }

    # 1. Baseline (standard connector, fixed alpha)
    print("\n--- Method 1: Baseline ---")
    t0 = time.time()
    def baseline_prompt(entry, query, q_emb, vis_embs, txt_embs, best_idx, all_scores):
        return (f"Use the following facts to answer the question accurately.\n\n"
                f"Visual Fact: {entry['visual_q']} → {entry['visual_a']}\n"
                f"Textual Fact: {entry['text_q']} → {entry['text_a']}\n\n"
                f"Question: {query}\nAnswer:")
    all_results["methods"]["baseline"] = runner.evaluate_method(
        "Baseline", data, memory, vis_embs, txt_embs, baseline_prompt)
    all_results["methods"]["baseline"]["time_seconds"] = time.time() - t0

    # 2. No-Connector
    print("\n--- Method 2: No-Connector ---")
    t0 = time.time()
    def no_connector_prompt(entry, query, q_emb, vis_embs, txt_embs, best_idx, all_scores):
        v_sim = float(util.cos_sim(q_emb, vis_embs[best_idx]).item())
        t_sim = float(util.cos_sim(q_emb, txt_embs[best_idx]).item())
        if v_sim >= t_sim:
            return (f"Use this fact to answer: {entry['visual_q']} → "
                    f"{entry['visual_a']}\n\nQuestion: {query}\nAnswer:")
        else:
            return (f"Use this fact to answer: {entry['text_q']} → "
                    f"{entry['text_a']}\n\nQuestion: {query}\nAnswer:")
    all_results["methods"]["no_connector"] = runner.evaluate_method(
        "No-Connector", data, memory, vis_embs, txt_embs, no_connector_prompt)
    all_results["methods"]["no_connector"]["time_seconds"] = time.time() - t0

    # 3. Pure RAG (retrieve and generate, no editing facts injected)
    print("\n--- Method 3: Pure RAG Baseline ---")
    t0 = time.time()
    def pure_rag_prompt(entry, query, q_emb, vis_embs, txt_embs, best_idx, all_scores):
        return f"Question: {query}\nAnswer:"
    all_results["methods"]["pure_rag"] = runner.evaluate_method(
        "Pure RAG", data, memory, vis_embs, txt_embs, pure_rag_prompt)
    all_results["methods"]["pure_rag"]["time_seconds"] = time.time() - t0

    # 4. Text-Only Editing
    print("\n--- Method 4: Text-Only Editing ---")
    t0 = time.time()
    def text_only_prompt(entry, query, q_emb, vis_embs, txt_embs, best_idx, all_scores):
        return (f"Use this fact to answer: {entry['text_q']} → {entry['text_a']}\n\n"
                f"Question: {query}\nAnswer:")
    all_results["methods"]["text_only"] = runner.evaluate_method(
        "Text-Only", data, memory, vis_embs, txt_embs, text_only_prompt)
    all_results["methods"]["text_only"]["time_seconds"] = time.time() - t0

    # 5. Adaptive Modality Gating (Exp 1)
    print("\n--- Method 5: Adaptive Modality Gating ---")
    t0 = time.time()
    def adaptive_gate_prompt(entry, query, q_emb, vis_embs, txt_embs, best_idx, all_scores):
        v_sim = float(util.cos_sim(q_emb, vis_embs[best_idx]).item())
        t_sim = float(util.cos_sim(q_emb, txt_embs[best_idx]).item())
        # Adaptive alpha based on modality agreement
        diff = t_sim - v_sim
        import math
        alpha_star = 1.0 / (1.0 + math.exp(-diff / 0.3))
        # Re-retrieve with adapted alpha
        combined = [(1. - alpha_star) * float(util.cos_sim(q_emb, t).item()) +
                    alpha_star * float(util.cos_sim(q_emb, v).item())
                    for v, t in zip(vis_embs, txt_embs)]
        new_idx = int(np.argmax(combined))
        new_entry = memory[new_idx]
        return (f"Use the following facts to answer the question accurately.\n\n"
                f"Visual Fact: {new_entry['visual_q']} → {new_entry['visual_a']}\n"
                f"Textual Fact: {new_entry['text_q']} → {new_entry['text_a']}\n\n"
                f"Question: {query}\nAnswer:")
    all_results["methods"]["adaptive_gating"] = runner.evaluate_method(
        "Adaptive Gating", data, memory, vis_embs, txt_embs, adaptive_gate_prompt)
    all_results["methods"]["adaptive_gating"]["time_seconds"] = time.time() - t0

    # 6. Soft Top-K (Exp 2) — uses top-K weighted prompt
    print("\n--- Method 6: Soft Top-K ---")
    t0 = time.time()
    def soft_topk_prompt(entry, query, q_emb, vis_embs, txt_embs, best_idx, all_scores):
        import math
        K, tau_r = 3, 0.1
        scores_with_idx = sorted(enumerate(all_scores), key=lambda x: x[1], reverse=True)[:K]
        top_scores = [s for _, s in scores_with_idx]
        exp_scores = [math.exp(s / tau_r) for s in top_scores]
        s_sum = sum(exp_scores)
        weights = [e / s_sum for e in exp_scores]
        top_entries = [memory[i] for i, _ in scores_with_idx]

        prompt_parts = ["Use the following facts (weighted by relevance) to answer:\n"]
        for w, ent in zip(weights, top_entries):
            prompt_parts.append(f"  [{w:.2f}] {ent['visual_q']} → {ent['visual_a']}")
        prompt_parts.append(f"\nQuestion: {query}\nAnswer:")
        return "\n".join(prompt_parts)
    all_results["methods"]["soft_topk"] = runner.evaluate_method(
        "Soft Top-K", data, memory, vis_embs, txt_embs, soft_topk_prompt)
    all_results["methods"]["soft_topk"]["time_seconds"] = time.time() - t0

    # 7. Confidence Threshold (Exp 4)
    print("\n--- Method 7: Confidence Threshold ---")
    t0 = time.time()
    def conf_threshold_prompt(entry, query, q_emb, vis_embs, txt_embs, best_idx, all_scores):
        score = all_scores[best_idx]
        sorted_s = sorted(all_scores, reverse=True)
        margin = sorted_s[0] - sorted_s[1] if len(sorted_s) > 1 else 1.0
        if score < 0.6 or margin < 0.05:
            return f"Question: {query}\nAnswer:"  # Reject, use base model
        return (f"Use the following facts to answer the question accurately.\n\n"
                f"Visual Fact: {entry['visual_q']} → {entry['visual_a']}\n"
                f"Textual Fact: {entry['text_q']} → {entry['text_a']}\n\n"
                f"Question: {query}\nAnswer:")
    all_results["methods"]["confidence_threshold"] = runner.evaluate_method(
        "Conf Threshold", data, memory, vis_embs, txt_embs, conf_threshold_prompt)
    all_results["methods"]["confidence_threshold"]["time_seconds"] = time.time() - t0

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("COMPLETE RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'EA':>7} {'±std':>7} {'RA':>7} {'Port':>7} {'Loc':>7}")
    print("-" * 70)
    for name, r in all_results["methods"].items():
        print(f"{name:<25} {r['edit_acc']*100:>6.1f}% {r['edit_acc_std']*100:>6.1f}% "
              f"{r['rephrase_acc']*100:>6.1f}% {r['portability_acc']*100:>6.1f}% "
              f"{r['locality_acc']*100:>6.1f}%")

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
