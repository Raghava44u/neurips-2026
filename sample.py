import json
import random
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

print("\n==============================")
print("🚀 MemEIC + BERT Retrieval")
print("==============================\n")

# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
DATA_PATH = "datasets/prompt/CCKEB_train.json"

with open(DATA_PATH, "r") as f:
    data = json.load(f)

samples = random.sample(data, 5)
print(f"✅ Loaded {len(samples)} samples\n")

# -------------------------------
# STEP 2: PARSE
# -------------------------------
def parse(sample):
    return {
        "image": sample["image"],
        "visual_q": sample["src"],
        "visual_a": sample["alt"],
        "text_q": sample["textual_edit"]["src"],
        "text_a": sample["textual_edit"]["alt"][0],
        "comp_q": sample["port_new"][0]["Q&A"]["Question"],
        "comp_ans": sample["port_new"][0]["Q&A"]["Answer"]
    }

memory = [parse(s) for s in samples]

# -------------------------------
# STEP 3: LOAD BERT MODEL
# -------------------------------
print("🔄 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings
memory_embeddings = []
for m in memory:
    text = m["visual_q"] + " " + m["text_q"]
    emb = embedder.encode(text, convert_to_tensor=True)
    memory_embeddings.append(emb)

print("✅ Embeddings ready\n")

# -------------------------------
# STEP 4: RETRIEVAL (BERT)
# -------------------------------
def retrieve(query):
    print("\n🔎 Query:", query)

    query_emb = embedder.encode(query, convert_to_tensor=True)

    scores = []
    for i, emb in enumerate(memory_embeddings):
        score = util.cos_sim(query_emb, emb).item()
        scores.append(score)
        print(f"Memory {i} similarity:", round(score, 3))

    best_idx = scores.index(max(scores))
    print("✅ Selected memory:", best_idx)

    return memory[best_idx]

# -------------------------------
# STEP 5: LOAD MODEL
# -------------------------------
MODEL = "microsoft/phi-2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)

model.to(device)
model.eval()

# -------------------------------
# STEP 6: GENERATE
# -------------------------------
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    return text.strip()

# -------------------------------
# STEP 7: TEST
# -------------------------------
for i, s in enumerate(memory):

    print("\n==============================")
    print(f"🧪 SAMPLE {i}")
    print("==============================")

    print("\n--- BEFORE ---")
    print(generate(s["comp_q"]))

    mem = retrieve(s["comp_q"])

    prompt = f"""
Use the facts to answer correctly.

Fact:
{mem['visual_q']} → {mem['visual_a']}

Fact:
{mem['text_q']} → {mem['text_a']}

Question: {s['comp_q']}
Answer:
"""

    print("\n--- AFTER ---")
    print(generate(prompt))

    print("✅ Expected:", s["comp_ans"])