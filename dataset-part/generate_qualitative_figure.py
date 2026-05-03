"""
Generate fig8_qualitative_analysis.png
Shows 5 representative examples (one per failure category) from Adversarial-2k.
For each example: image | question | baseline prediction (wrong) | GC prediction (correct).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import textwrap

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "..")
IMG_ROOT = os.path.join(REPO, "datasets", "CCKEB_images", "mmkb_images")
OUT_PATH = os.path.join(BASE, "new-figs", "fig8_qualitative_analysis.png")

# ── 5 handpicked examples (one per failure category) ─────────────────────────
# Fields: category label, image path (relative to IMG_ROOT), question,
#         baseline_pred (WRONG), gc_pred (CORRECT / ground truth)
EXAMPLES = [
    {
        "cat": "F1 · Polysemy",
        "color": "#3498db",
        "img": "m.01b7h8/google_2.jpg",
        "question": "What palm appears\nin this photograph?",
        "baseline": "Palm reading",
        "gc": "Coconut palm tree",
        "note": "Textual edit activates palmistry sense;\nvisual context ignored by baseline.",
    },
    {
        "cat": "F2 · Cross-Modal Conflict",
        "color": "#e74c3c",
        "img": "m.0n7q7/bing_2.jpg",
        "question": "What is this bark\nin the image?",
        "baseline": "Oak tree bark",
        "gc": "Dog barking",
        "note": "Textual edit to botany sense of 'bark'\noverrides unambiguous visual (dog).",
    },
    {
        "cat": "F3 · Near-Miss Retrieval",
        "color": "#e67e22",
        "img": "m.03cx282/google_0.jpg",
        "question": "About microscope development:\nwhat did he discover?",
        "baseline": "Antonie van Leeuwenhoek",
        "gc": "Microorganisms",
        "note": "Correct memory block retrieved but\nwrong field (entity vs. discovery).",
    },
    {
        "cat": "F4 · Multi-Hop Reasoning",
        "color": "#8e44ad",
        "img": "m.0gh65c5/google_3.jpg",
        "question": "About Braille writing\nin this image: at what age?",
        "baseline": "Louis Braille",
        "gc": "15 years old",
        "note": "Chain image→invention→inventor→age;\nbaseline halts at hop 2.",
    },
    {
        "cat": "F5 · Hard Visual Distinction",
        "color": "#27ae60",
        "img": "m.0ct2tf5/bing_23.jpg",
        "question": "Identify the exact\nflat-faced cat in this picture.",
        "baseline": "Himalayan cat\n(Points pattern)",
        "gc": "Persian cat\n(Solid colors)",
        "note": "Fine-grained breed confusion;\ntext alone cannot disambiguate.",
    },
]

# ── Layout ───────────────────────────────────────────────────────────────────
N = len(EXAMPLES)
fig = plt.figure(figsize=(18, 3.8 * N), facecolor="#fafafa")
fig.patch.set_facecolor("#fafafa")

outer = GridSpec(N, 1, figure=fig, hspace=0.06)

for row, ex in enumerate(EXAMPLES):
    inner = outer[row].subgridspec(1, 4, wspace=0.05,
                                   width_ratios=[1, 1.4, 1.4, 1.6])

    # ── (A) Image ──────────────────────────────────────────────────────────
    ax_img = fig.add_subplot(inner[0])
    img_path = os.path.join(IMG_ROOT, ex["img"].replace("/", os.sep))
    try:
        img = Image.open(img_path).convert("RGB")
        ax_img.imshow(img, aspect="auto")
    except Exception:
        ax_img.text(0.5, 0.5, "[image]", ha="center", va="center",
                    transform=ax_img.transAxes, fontsize=10, color="gray")
    ax_img.axis("off")
    # Category label as colored top border
    ax_img.set_title(ex["cat"], fontsize=11, fontweight="bold",
                     color="white",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=ex["color"],
                               edgecolor="none"),
                     pad=5)

    # ── (B) Question card ──────────────────────────────────────────────────
    ax_q = fig.add_subplot(inner[1])
    ax_q.set_facecolor("#ecf0f1")
    ax_q.set_xlim(0, 1); ax_q.set_ylim(0, 1)
    ax_q.axis("off")
    ax_q.text(0.5, 0.88, "Query", ha="center", va="top",
              fontsize=9, fontweight="bold", color="#555")
    ax_q.text(0.5, 0.62, ex["question"], ha="center", va="top",
              fontsize=10.5, color="#222",
              multialignment="center",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor="#bdc3c7", linewidth=1.2))
    ax_q.text(0.5, 0.12, ex["note"], ha="center", va="bottom",
              fontsize=7.5, color="#7f8c8d",
              style="italic", multialignment="center")

    # ── (C) Baseline prediction (WRONG) ────────────────────────────────────
    ax_b = fig.add_subplot(inner[2])
    ax_b.set_facecolor("#fef9f9")
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1)
    ax_b.axis("off")
    ax_b.text(0.5, 0.90, "Baseline  ✗", ha="center", va="top",
              fontsize=9, fontweight="bold", color="#c0392b")
    ax_b.text(0.5, 0.55, ex["baseline"], ha="center", va="center",
              fontsize=13, fontweight="bold", color="#c0392b",
              multialignment="center",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#fdecea",
                        edgecolor="#e74c3c", linewidth=2))
    ax_b.text(0.5, 0.14, "MemEIC base\n(EA ≈ 0.459)", ha="center", va="bottom",
              fontsize=7.5, color="#999", multialignment="center")

    # ── (D) GC prediction (CORRECT) ────────────────────────────────────────
    ax_gc = fig.add_subplot(inner[3])
    ax_gc.set_facecolor("#f9fef9")
    ax_gc.set_xlim(0, 1); ax_gc.set_ylim(0, 1)
    ax_gc.axis("off")
    ax_gc.text(0.5, 0.90, "MemEIC + GC  ✓", ha="center", va="top",
               fontsize=9, fontweight="bold", color="#27ae60")
    ax_gc.text(0.5, 0.55, ex["gc"], ha="center", va="center",
               fontsize=13, fontweight="bold", color="#27ae60",
               multialignment="center",
               bbox=dict(boxstyle="round,pad=0.5", facecolor="#eafaf1",
                         edgecolor="#2ecc71", linewidth=2))
    ax_gc.text(0.5, 0.14, "Gated Connector variant\n(EA ≈ 0.514, +5.5 pp)",
               ha="center", va="bottom",
               fontsize=7.5, color="#999", multialignment="center")

# ── Global title & legend ────────────────────────────────────────────────────
fig.suptitle(
    "Qualitative Analysis: Adversarial-2k Failure Cases\n"
    "Baseline (MemEIC) vs. Best Variant (MemEIC + Gated Connector)",
    fontsize=14, fontweight="bold", y=1.005, color="#2c3e50"
)

# Column headers on first row only
for ax, label in zip(
    [fig.axes[0], fig.axes[1], fig.axes[2], fig.axes[3]],
    ["Image", "Query", "Baseline Prediction\n(Wrong)", "GC Prediction\n(Correct)"]
):
    ax.annotate(
        label, xy=(0.5, 1.0), xycoords="axes fraction",
        ha="center", va="bottom", fontsize=9, color="#555",
        style="italic"
    )

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {OUT_PATH}")
