# How to Overcome MemEIC Limitations: Concrete Solutions

**Based on**: 200-sample adversarial failure analysis  
**Target file**: `easyeditor/trainer/algs/OURS.py` (primary), `blip2_opt.py` (secondary)

---

## Summary of 5 Identified Limitations → 5 Solutions

| # | Limitation | Root Cause (Code Location) | Proposed Solution |
|---|-----------|---------------------------|-------------------|
| 1 | **Polysemy Blindness** | Hard K-NN `.max(-1)` selects single example (OURS.py L716-727) | Soft Top-K retrieval with confidence weighting |
| 2 | **Modality Conflict** | Fixed 0.1×text + 0.9×image weighting (OURS.py L724) | Adaptive gating with conflict detection |
| 3 | **Connector Hurts Performance** | Only 3 LoRA steps on q_proj/k_proj (OURS.py L505, blip2_opt.py L125) | Consistency-checked connector with contrastive loss |
| 4 | **Retrieval Errors / No Rejection** | Always picks best match, no threshold (OURS.py L716) | Confidence threshold + fallback to base model |
| 5 | **Portability Collapse (14%)** | No knowledge propagation mechanism | Semantic neighborhood expansion during editing |

---

## Solution 1: Soft Top-K Retrieval (Fix Polysemy Blindness)

### Problem
```python
# OURS.py Line 716 — picks ONE best match, ignores alternatives
cls_sims, cls_idxs = sims.max(-1)  # Hard K-NN selection
```
When "crane" appears, it picks either bird OR machine — never considers both.

### Solution: Weighted Top-K with Attention
Instead of hard `.max(-1)`, retrieve **top-K candidates** and use attention-weighted aggregation:

```python
# REPLACE hard selection with soft top-K
def soft_topk_retrieval(self, sims, k=3, temperature=0.1):
    """
    Retrieve top-K candidates and weight them by softmax similarity.
    Returns weighted index and aggregated confidence.
    """
    topk_sims, topk_idxs = sims.topk(min(k, sims.shape[-1]), dim=-1)
    
    # Softmax weighting — high temperature = more uniform, low = more peaked
    weights = torch.softmax(topk_sims / temperature, dim=-1)
    
    # If top-1 dominates (>0.8 weight), use it directly (confident match)
    # If not, flag ambiguity for downstream handling
    top1_weight = weights[..., 0]
    is_ambiguous = top1_weight < 0.7  # ambiguity flag
    
    # Weighted confidence
    weighted_sim = (weights * topk_sims).sum(-1)
    best_idx = topk_idxs[..., 0]  # still return top-1 as primary
    
    return weighted_sim, best_idx, topk_idxs, weights, is_ambiguous
```

### Where to Apply
Replace **3 locations** in `OURS.py`:
- `run_image_classifier()` Line 716: `cls_sims, cls_idxs = sims.max(-1)`
- `run_image_classifier()` Line 727: `cls_sims, cls_idxs = final_sims.max(-1)`
- `run_text_classifier()` Line 744: `cls_sims, cls_idxs = sims.max(-1)`

### Impact
- Detects ambiguous queries (crane, bank, bark) via `is_ambiguous` flag
- Enables downstream logic to request visual disambiguation
- Falls back to current behavior on unambiguous queries (top-1 weight > 0.7)

---

## Solution 2: Adaptive Modality Gating (Fix Modality Conflict)

### Problem  
```python
# OURS.py Line 724 — fixed weights ignore when modalities disagree
final_sims = (0.1 * sims + 0.9 * image_sims)
```
Image always dominates 90%. When image says "apple fruit" but text says "Apple Inc", image wins blindly.

### Solution: Learned Gating with Conflict Detection

```python
def adaptive_modality_fusion(self, text_sims, image_sims, threshold=0.3):
    """
    Dynamically weight text vs image based on agreement.
    When modalities agree → use image-heavy weighting (default behavior)
    When modalities conflict → reduce confidence, weight more equally
    """
    # Detect conflict: do text and image point to DIFFERENT cache entries?
    text_best = text_sims.argmax(-1)
    image_best = image_sims.argmax(-1)
    modalities_agree = (text_best == image_best)
    
    # Measure disagreement magnitude
    # If text says idx=3 strongly but image says idx=7 strongly → conflict
    text_confidence = text_sims.max(-1).values - text_sims.median(-1).values
    image_confidence = image_sims.max(-1).values - image_sims.median(-1).values
    
    # Adaptive alpha: when agreeing, use 0.9 image; when conflicting, use 0.5/0.5
    alpha_image = torch.where(
        modalities_agree,
        torch.tensor(0.9, device=text_sims.device),   # agreement → trust image
        torch.tensor(0.5, device=text_sims.device)     # conflict → equal weight
    )
    alpha_text = 1.0 - alpha_image
    
    final_sims = alpha_text * text_sims + alpha_image * image_sims
    
    # Flag conflicts for logging/handling
    has_conflict = ~modalities_agree
    
    return final_sims, has_conflict
```

### Where to Apply
Replace in `run_image_classifier()` Line 724:
```python
# OLD:
final_sims = (0.1 * sims + 0.9 * image_sims)
# NEW:
final_sims, has_conflict = self.adaptive_modality_fusion(sims, image_sims)
```

### Impact
- Prevents blind image dominance when text disagrees
- Conflict flag enables downstream disambiguation
- No performance penalty on agreeing cases (preserves default behavior)

---

## Solution 3: Consistency-Checked Connector (Fix Connector Paradox)

### Problem
The connector trains for **only 3 steps** (OURS.py L505: `if connector_mode and it >= 2: break`) on `q_proj`/`k_proj` with no consistency check. Under adversarial conditions, it composes **incompatible** visual + textual knowledge.

Current ablation showed: **removing the connector improves accuracy** (71% → 86.5%).

### Solution A: Contrastive Consistency Loss

Add a consistency loss term that penalizes the connector when its composed output contradicts either the visual-only or textual-only prediction:

```python
def connector_consistency_loss(self, composed_logits, visual_logits, textual_logits, labels):
    """
    Ensure connector output is consistent with individual modality outputs.
    Penalize if connector predicts something NEITHER visual nor textual predicted.
    """
    # Standard edit loss
    edit_loss = F.cross_entropy(
        composed_logits.view(-1, composed_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # Consistency: composed output should be close to at least one modality
    visual_probs = F.softmax(visual_logits, dim=-1).detach()
    textual_probs = F.softmax(textual_logits, dim=-1).detach()
    composed_probs = F.log_softmax(composed_logits, dim=-1)
    
    # KL divergence to closest modality (min of two KLs)
    kl_visual = F.kl_div(composed_probs, visual_probs, reduction='batchmean')
    kl_textual = F.kl_div(composed_probs, textual_probs, reduction='batchmean')
    consistency_loss = torch.min(kl_visual, kl_textual)
    
    total_loss = edit_loss + 0.5 * consistency_loss
    return total_loss
```

### Solution B: Increase Connector Training Steps

The connector gets only 3 optimization steps vs 10 for visual/textual adapters. This is insufficient for learning proper cross-modal alignment:

```python
# OURS.py Line 505 — increase from 3 to 8 steps
if connector_mode and it >= 7:  # was: it >= 2 (3 steps)
    break
```

### Solution C: Gated Connector (Bypass When Uncertain)

Add a learned gate that can **bypass the connector** when it detects it would hurt:

```python
# In the connector forward pass, add a confidence gate
connector_gate = torch.sigmoid(self.gate_linear(torch.cat([visual_repr, textual_repr], dim=-1)))
# gate ∈ [0,1]: 0 = skip connector, 1 = fully use connector
# During adversarial inputs, gate learns to reduce connector influence
output = connector_gate * composed_output + (1 - connector_gate) * fallback_output
```

---

## Solution 4: Retrieval Confidence Threshold (Fix Retrieval Errors)

### Problem
The system **always** returns a match, even when nothing in cache is relevant. High confidence (0.96) doesn't mean correct answer.

### Solution: Confidence-Based Rejection

```python
def run_image_classifier_with_threshold(self, *inputs, min_confidence=0.6, **kwargs):
    """Modified classifier that can REJECT low-confidence matches."""
    cls_sims, cls_idxs, log_sim_matrix = self.run_image_classifier(*inputs, **kwargs)
    
    if cls_sims is not None:
        # Check if best match exceeds confidence threshold
        confident = cls_sims > min_confidence
        
        if not confident.all():
            # For low-confidence cases, return None → triggers base model fallback
            # This prevents injecting wrong knowledge when retrieval is uncertain
            cls_sims = torch.where(confident, cls_sims, torch.zeros_like(cls_sims))
            cls_idxs = torch.where(confident, cls_idxs, torch.full_like(cls_idxs, -1))
    
    return cls_sims, cls_idxs, log_sim_matrix

# In the main edit/test pipeline, handle rejection:
def generate_with_fallback(self, cls_sims, cls_idxs, base_model, query):
    if cls_idxs == -1:  # rejected by retrieval
        # Fall back to base model without any knowledge injection
        return base_model.generate(query)
    else:
        # Use retrieved knowledge as normal
        return self.generate_with_retrieval(cls_idxs, query)
```

### Additional Fix: Relative Confidence Check

```python
def relative_confidence_check(self, sims):
    """
    Check if top match is significantly better than 2nd best.
    If top-1 and top-2 are close, the match is ambiguous.
    """
    if sims.shape[-1] < 2:
        return True  # only one candidate
    
    sorted_sims = sims.sort(-1, descending=True).values
    margin = sorted_sims[..., 0] - sorted_sims[..., 1]
    
    # If margin < 0.1, top two matches are too close → ambiguous
    is_confident = margin > 0.1
    return is_confident
```

---

## Solution 5: Semantic Neighborhood Expansion (Fix Portability Collapse)

### Problem
Portability stays at 14% regardless of retrieval weight α. Edits are memorized verbatim but never generalize to semantically related queries.

**Root cause**: When you edit "The crane shown is a Sandhill Crane", only the exact query "What type of crane is shown?" is cached. Paraphrased queries like "What bird species is this?" have no matching cache entry.

### Solution A: Query Augmentation During Editing

When a new fact is edited into memory, **generate paraphrases** and store them all:

```python
def store_edit_with_augmentation(self, question, answer, image, num_augments=3):
    """
    Store the original edit PLUS paraphrased versions to improve portability.
    """
    # Store original
    self.cache_inputs.append(question)
    self.cache_labels.append(answer)
    
    # Generate paraphrases using the base LLM itself
    paraphrase_prompt = f"Rephrase this question in {num_augments} different ways:\n'{question}'\n1."
    paraphrases = self.model.generate(paraphrase_prompt, num_return_sequences=num_augments)
    
    # Store paraphrases with same answer
    for para in paraphrases:
        self.cache_inputs.append(para)
        self.cache_labels.append(answer)
        self.cache_image_inputs.append(image)
    
    # Also store a "semantic key" — the core concept
    # E.g., "crane shown" → embed as concept, not just exact string
    concept_key = self.extract_concept(question)  # e.g., "crane identification"
    self.cache_inputs.append(concept_key)
    self.cache_labels.append(answer)
```

### Solution B: Embedding Neighborhood Retrieval

Instead of exact query matching, match against a **neighborhood** in embedding space:

```python
def embedding_logsim_matrix_with_neighborhood(self, cls_ctxs, test_input_text, neighborhood_radius=0.15):
    """
    Modified similarity function that considers semantic neighborhoods.
    If test query is within radius of ANY cached query, retrieve that knowledge.
    """
    # Get standard embeddings
    log_sims = self.embedding_logsim_matrix(cls_ctxs, test_input_text)
    sims = log_sims.exp()
    
    # Instead of hard max, find ALL entries within neighborhood
    within_neighborhood = sims > (sims.max(-1, keepdim=True).values - neighborhood_radius)
    
    # Aggregate answers from neighborhood (majority voting or confidence-weighted)
    neighborhood_sims = sims * within_neighborhood.float()
    
    return neighborhood_sims.log()
```

### Solution C: Portability-Aware Training Loss

Add a training objective that explicitly optimizes for portability:

```python
def portability_loss(self, model, edit_fact, rephrase_questions, original_answer):
    """
    During training, also optimize that rephrased questions get the same answer.
    This teaches the model to generalize edits, not just memorize them.
    """
    losses = []
    for rephrase in rephrase_questions:
        output = model.generate(rephrase)
        loss = F.cross_entropy(output.logits, original_answer)
        losses.append(loss)
    
    return torch.stack(losses).mean()

# Add to total training loss:
# total_loss = edit_loss + locality_loss + 0.5 * portability_loss
```

---

## Implementation Priority

| Priority | Solution | Effort | Expected Impact |
|----------|----------|--------|-----------------|
| **P0** | Solution 4: Confidence Threshold | Low (10 lines) | Stops 40 retrieval errors immediately |
| **P1** | Solution 2: Adaptive Gating | Low (20 lines) | Fixes 40 modality conflict cases |
| **P2** | Solution 1: Soft Top-K | Medium (30 lines) | Fixes 47 ambiguity cases |
| **P3** | Solution 3: Connector Fix | Medium (50 lines) | Fixes connector paradox (+15.5% accuracy) |
| **P4** | Solution 5: Portability | High (100+ lines) | Addresses 14% portability collapse |

---

## Quick Win: Minimum Viable Fix (3 Lines Changed)

The smallest change with the biggest impact — fix the modality weighting:

```python
# In run_image_classifier(), Line 724 of OURS.py
# BEFORE:
final_sims = (0.1 * sims + 0.9 * image_sims)

# AFTER (detect conflict, use equal weighting when modalities disagree):
agree = (sims.argmax(-1) == image_sims.argmax(-1)).float()
alpha = 0.9 * agree + 0.5 * (1 - agree)  # 0.9 if agree, 0.5 if conflict
final_sims = ((1 - alpha) * sims + alpha * image_sims)
```

This alone should fix ~20% of the conflicting signal failures with zero risk to normal operation.

---

## Architecture-Level Recommendations

| Current Design | Problem | Recommended Change |
|---------------|---------|-------------------|
| Single DistilBERT for text similarity | Different embedding space from CLIP | Use a **unified multimodal encoder** (e.g., BLIP-2's own Q-Former for both) |
| Separate visual/textual LoRA adapters | No shared representation | Add **shared LoRA layers** that both modalities pass through |
| 3-step connector training | Insufficient for cross-modal alignment | Train connector for **8-10 steps** with consistency loss |
| Cache stores raw Q/A pairs | No semantic generalization | Store **concept-level embeddings** alongside raw text |
| L2 distance in anisotropic space | Poor similarity estimation | Switch to **cosine similarity** (`cos: True` in hparams) |

---

*These solutions are designed to be applied incrementally. Start with P0 (confidence threshold) and measure impact before proceeding to more complex changes.*
