"""
Visual Failure Demonstration for MemEIC
========================================
Generates publication-quality figures showing:
  1. A grid of 6 severe failure cases with images + Q/A annotations
  2. Side-by-side correct vs wrong comparison panel  
  3. Category-wise failure heatmap
  4. Retrieval confidence vs actual correctness scatter
"""
import json, os, textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
import numpy as np

os.makedirs('results/plots', exist_ok=True)

IMG_BASE = 'datasets/CCKEB_images/mmkb_images'
fc = json.load(open('results/failure_cases.json'))
adv = json.load(open('datasets/adversarial_v2_hard.json'))
orig_data = json.load(open('datasets/CCKEB_eval.json'))

plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
})

# ============================================================
# FIGURE 1: 6-Panel Failure Gallery (the main visual demo)
# ============================================================
print("Generating Figure 1: Visual Failure Gallery...")

# Hand-picked 6 diverse, visually compelling failures
selected_idxs = [32, 22, 50, 134, 61, 153]
# crane(ambiguity), rock(conflict), metal(hard), bark(ambiguity), tree(conflict), instrument(conflict)

fig, axes = plt.subplots(2, 3, figsize=(18, 13))
fig.suptitle('MemEIC Visual Reasoning Failures: The Model Sees But Does Not Understand',
             fontsize=17, fontweight='bold', y=0.98)

for ax_idx, sample_idx in enumerate(selected_idxs):
    row, col = divmod(ax_idx, 3)
    ax = axes[row][col]

    # Get failure case and dataset entry
    case = next(c for c in fc if c['sample_idx'] == sample_idx)
    s = adv[sample_idx]
    img_path = os.path.join(IMG_BASE, s['image'])

    # Load and display image
    try:
        img = mpimg.imread(img_path)
        ax.imshow(img)
    except Exception as e:
        ax.text(0.5, 0.5, f'Image\nUnavailable', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')

    ax.set_xticks([])
    ax.set_yticks([])

    # Category badge color
    cat_colors = {
        'ambiguity': '#e74c3c',
        'conflicting_signals': '#f39c12',
        'hard_distinction': '#9b59b6',
        'reasoning_failure': '#3498db',
        'retrieval_error': '#1abc9c',
    }
    badge_color = cat_colors.get(case['failure_type'], '#95a5a6')
    cat_label = case['failure_type'].replace('_', ' ').title()

    # Question
    q_wrapped = textwrap.fill(case['question'], width=45)
    ax.set_title(f"Q: {q_wrapped}", fontsize=11, fontweight='bold', pad=8, loc='left')

    # Expected vs Predicted annotation below image
    exp_text = textwrap.fill(case['expected'], width=40)
    pred_text = textwrap.fill(case['predicted'], width=40)

    # Category badge
    ax.text(0.02, 0.96, f" {cat_label} ", transform=ax.transAxes,
            fontsize=9, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color, alpha=0.9),
            va='top', ha='left')

    # Fails badge
    ax.text(0.98, 0.96, f" {case['num_failures']} metrics failed ",
            transform=ax.transAxes, fontsize=9, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#c0392b', alpha=0.9),
            va='top', ha='right')

    # Red/Green answer boxes below image
    box_y = -0.02
    ax.text(0.0, box_y, f"✓ Expected: {exp_text}",
            transform=ax.transAxes, fontsize=10, color='#27ae60', fontweight='bold',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#eafaf1', edgecolor='#27ae60', linewidth=1.5))
    ax.text(0.0, box_y - 0.14, f"✗ MemEIC: {pred_text}",
            transform=ax.transAxes, fontsize=10, color='#e74c3c', fontweight='bold',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdedec', edgecolor='#e74c3c', linewidth=1.5))

plt.subplots_adjust(hspace=0.45, wspace=0.15, top=0.92, bottom=0.08)
plt.savefig('results/plots/visual_failure_gallery.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("  Saved: results/plots/visual_failure_gallery.png")


# ============================================================
# FIGURE 2: Side-by-side — What image shows vs What model says
# ============================================================
print("Generating Figure 2: Image vs Model Prediction Comparison...")

# Pick 4 cases that are very visual (object identification failures)
visual_cases = [32, 50, 61, 134]  # crane, metal, tree, bark

fig, axes = plt.subplots(2, 4, figsize=(20, 10),
                          gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('What the Image Shows vs What MemEIC Predicts',
             fontsize=17, fontweight='bold', y=0.98)

for col_idx, sample_idx in enumerate(visual_cases):
    case = next(c for c in fc if c['sample_idx'] == sample_idx)
    s = adv[sample_idx]
    img_path = os.path.join(IMG_BASE, s['image'])

    # Image row
    ax_img = axes[0][col_idx]
    try:
        img = mpimg.imread(img_path)
        ax_img.imshow(img)
    except:
        ax_img.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_img.transAxes)
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    q_short = case['question'][:50]
    ax_img.set_title(f"Q: {q_short}", fontsize=11, fontweight='bold', pad=6)

    # Red border to indicate failure
    for spine in ax_img.spines.values():
        spine.set_edgecolor('#e74c3c')
        spine.set_linewidth(3)
        spine.set_visible(True)

    # Text row below
    ax_txt = axes[1][col_idx]
    ax_txt.axis('off')

    exp_wrapped = textwrap.fill(case['expected'], width=28)
    pred_wrapped = textwrap.fill(case['predicted'], width=28)

    ax_txt.text(0.5, 0.85, f"✓ Correct:", ha='center', fontsize=11,
                fontweight='bold', color='#27ae60', transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.6, exp_wrapped, ha='center', fontsize=10,
                color='#27ae60', transform=ax_txt.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#27ae60'))

    ax_txt.text(0.5, 0.3, f"✗ MemEIC:", ha='center', fontsize=11,
                fontweight='bold', color='#e74c3c', transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.05, pred_wrapped, ha='center', fontsize=10,
                color='#e74c3c', transform=ax_txt.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdedec', edgecolor='#e74c3c'))

plt.subplots_adjust(hspace=0.05, wspace=0.2, top=0.91, bottom=0.02)
plt.savefig('results/plots/visual_image_vs_prediction.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("  Saved: results/plots/visual_image_vs_prediction.png")


# ============================================================
# FIGURE 3: Retrieval Confidence vs Correctness (Scatter)
# Shows model is CONFIDENT but WRONG
# ============================================================
print("Generating Figure 3: Confidence vs Correctness Scatter...")

fig, ax = plt.subplots(figsize=(10, 7))

# Categorize all samples
for c in fc:
    marker = 'o'
    color = '#e74c3c'  # red = failure
    size = 30 + c['num_failures'] * 15
    alpha = 0.6

    cat_colors_scatter = {
        'ambiguity': '#e74c3c',
        'conflicting_signals': '#f39c12',
        'hard_distinction': '#9b59b6',
        'reasoning_failure': '#3498db',
        'retrieval_error': '#1abc9c',
    }
    color = cat_colors_scatter.get(c['failure_type'], '#95a5a6')
    ax.scatter(c['retrieval_score'], c['num_failures'], c=color, s=size,
              alpha=alpha, edgecolors='black', linewidth=0.3)

# Add category legend manually
for cat, color in cat_colors_scatter.items():
    ax.scatter([], [], c=color, s=80, label=cat.replace('_', ' ').title(),
              edgecolors='black', linewidth=0.5)

# Highlight the "confident but wrong" zone
ax.axvspan(0.8, 1.0, alpha=0.08, color='red')
ax.text(0.90, 4.7, 'HIGH CONFIDENCE\nBUT WRONG', ha='center', fontsize=12,
        fontweight='bold', color='#c0392b', alpha=0.8)

ax.axhline(y=3, color='gray', linestyle='--', alpha=0.4)
ax.text(0.52, 3.1, 'Severe failure threshold', fontsize=9, color='gray', alpha=0.7)

ax.set_xlabel('Retrieval Confidence Score', fontsize=14)
ax.set_ylabel('Number of Metrics Failed', fontsize=14)
ax.set_title('MemEIC Retrieval Confidence vs Actual Failure Severity\n'
             '(High confidence ≠ correct answer)', fontsize=15, fontweight='bold')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0.5, 5.5)
plt.tight_layout()
plt.savefig('results/plots/visual_confidence_vs_failure.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved: results/plots/visual_confidence_vs_failure.png")


# ============================================================
# FIGURE 4: Failure Category Breakdown — Where Visual Reasoning Fails
# ============================================================
print("Generating Figure 4: Failure Category Breakdown...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: category counts as horizontal bar
cats = {}
for c in fc:
    ft = c['failure_type'].replace('_', ' ').title()
    if ft not in cats:
        cats[ft] = {'count': 0, 'total_fails': 0}
    cats[ft]['count'] += 1
    cats[ft]['total_fails'] += c['num_failures']

cat_names = list(cats.keys())
cat_counts = [cats[c]['count'] for c in cat_names]
cat_avg_fails = [cats[c]['total_fails'] / cats[c]['count'] for c in cat_names]

# Sort by count
sort_idx = np.argsort(cat_counts)
cat_names_sorted = [cat_names[i] for i in sort_idx]
cat_counts_sorted = [cat_counts[i] for i in sort_idx]
cat_avg_sorted = [cat_avg_fails[i] for i in sort_idx]

bar_colors = ['#9b59b6', '#3498db', '#1abc9c', '#f39c12', '#e74c3c']
bars = ax1.barh(cat_names_sorted, cat_counts_sorted, color=bar_colors, edgecolor='white', height=0.6)
for bar, count in zip(bars, cat_counts_sorted):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{count}', va='center', fontsize=12, fontweight='bold')

ax1.set_xlabel('Number of Failure Cases', fontsize=13)
ax1.set_title('Failure Cases by Category', fontsize=14, fontweight='bold')
ax1.set_xlim(0, max(cat_counts_sorted) + 5)

# Right: average severity per category
bars2 = ax2.barh(cat_names_sorted, cat_avg_sorted, color=bar_colors, edgecolor='white', height=0.6)
for bar, avg in zip(bars2, cat_avg_sorted):
    ax2.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
            f'{avg:.2f}', va='center', fontsize=12, fontweight='bold')

ax2.set_xlabel('Avg Metrics Failed per Sample', fontsize=13)
ax2.set_title('Avg Failure Severity by Category', fontsize=14, fontweight='bold')
ax2.set_xlim(0, max(cat_avg_sorted) + 0.5)
ax2.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='Severe threshold')
ax2.legend(fontsize=10)

plt.suptitle('Where MemEIC Visual Reasoning Breaks Down',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/plots/visual_failure_categories.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved: results/plots/visual_failure_categories.png")


# ============================================================
# FIGURE 5: Failure Mode Explanation Infographic
# ============================================================
print("Generating Figure 5: Failure Mode Infographic...")

fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle('Three Critical Visual Reasoning Failure Modes in MemEIC',
             fontsize=17, fontweight='bold', y=0.98)

failure_modes = [
    {
        'title': 'Polysemy Blindness',
        'sample_idx': 32,
        'explanation': (
            'The model cannot disambiguate\n'
            'polysemous words using visual\n'
            'context. "Crane" defaults to\n'
            'construction equipment despite\n'
            'the image showing a bird.'
        ),
        'color': '#e74c3c',
    },
    {
        'title': 'Modality Conflict',
        'sample_idx': 22,
        'explanation': (
            'When image evidence conflicts\n'
            'with retrieved text, the model\n'
            'ignores visual features. Sees\n'
            'sedimentary limestone but\n'
            'predicts "rapid cooling of lava".'
        ),
        'color': '#f39c12',
    },
    {
        'title': 'Hallucinated Properties',
        'sample_idx': 50,
        'explanation': (
            'The model confabulates fine-\n'
            'grained properties it cannot\n'
            'actually see. Predicts "aluminum"\n'
            'when the actual alloy is\n'
            '"304 Stainless Steel".'
        ),
        'color': '#9b59b6',
    },
]

for ax_idx, mode in enumerate(failure_modes):
    ax = axes[ax_idx]
    case = next(c for c in fc if c['sample_idx'] == mode['sample_idx'])
    s = adv[mode['sample_idx']]
    img_path = os.path.join(IMG_BASE, s['image'])

    # Create sub-axes: image top, text bottom
    # Use the full ax for layout
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.8, mode['title'], ha='center', fontsize=15, fontweight='bold',
            color=mode['color'])

    # Image in the middle area
    img_ax = fig.add_axes([
        ax.get_position().x0 + 0.02,
        ax.get_position().y0 + 0.35,
        ax.get_position().width - 0.04,
        ax.get_position().height * 0.45
    ])
    try:
        img = mpimg.imread(img_path)
        img_ax.imshow(img)
    except:
        img_ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=img_ax.transAxes)
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    for spine in img_ax.spines.values():
        spine.set_edgecolor(mode['color'])
        spine.set_linewidth(3)
        spine.set_visible(True)

    # Question
    q_text = textwrap.fill(f"Q: {case['question']}", width=35)
    ax.text(5, 5.5, q_text, ha='center', fontsize=10, fontweight='bold',
            style='italic')

    # Expected / Predicted
    ax.text(5, 4.5, f"✓ {case['expected'][:40]}", ha='center', fontsize=10,
            color='#27ae60', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#27ae60'))
    ax.text(5, 3.5, f"✗ {case['predicted'][:40]}", ha='center', fontsize=10,
            color='#e74c3c', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdedec', edgecolor='#e74c3c'))

    # Explanation
    ax.text(5, 1.5, mode['explanation'], ha='center', fontsize=10,
            color='#2c3e50', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor=mode['color'],
                      linewidth=2))

plt.savefig('results/plots/visual_failure_modes.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("  Saved: results/plots/visual_failure_modes.png")


# ============================================================
# FIGURE 6: Before/After — How Correct Edit Looks vs MemEIC
# ============================================================
print("Generating Figure 6: Correct Location vs MemEIC Failure...")

# Show what a correct knowledge edit chain looks like vs where MemEIC breaks
fig = plt.figure(figsize=(18, 9))
fig.suptitle('Knowledge Edit Pipeline: Where MemEIC Fails in the Chain',
             fontsize=17, fontweight='bold', y=0.98)

# Create a flow diagram with 3 columns showing the pipeline
# Column 1: Input (Image + Question)
# Column 2: Retrieval step
# Column 3: Output (Correct vs Wrong)

# Use 3 rows for 3 examples
examples = [
    {'idx': 32, 'step_fail': 'Visual Understanding',
     'explain': 'Retrieves "crane" facts but picks\nconstruction crane, not bird'},
    {'idx': 102, 'step_fail': 'Cross-Modal Fusion',
     'explain': 'Sees bamboo plant but connector\noutputs furniture/flooring usage'},
    {'idx': 64, 'step_fail': 'Answer Generation',
     'explain': 'Retrieves mineral info but\ngenerates country names instead'},
]

for row_idx, ex in enumerate(examples):
    case = next(c for c in fc if c['sample_idx'] == ex['idx'])
    s = adv[ex['idx']]
    img_path = os.path.join(IMG_BASE, s['image'])

    y_start = 0.68 - row_idx * 0.32

    # Column 1: Image
    img_ax = fig.add_axes([0.02, y_start, 0.18, 0.25])
    try:
        img = mpimg.imread(img_path)
        img_ax.imshow(img)
    except:
        pass
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    img_ax.set_title(f"Q: {case['question'][:35]}...", fontsize=9, fontweight='bold', pad=4)
    for spine in img_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('#3498db')

    # Column 2: Pipeline step that fails
    txt_ax = fig.add_axes([0.25, y_start, 0.25, 0.25])
    txt_ax.axis('off')
    txt_ax.text(0.5, 0.8, f"Fails at: {ex['step_fail']}", ha='center',
                fontsize=12, fontweight='bold', color='#e74c3c',
                transform=txt_ax.transAxes)
    txt_ax.text(0.5, 0.5, ex['explain'], ha='center', fontsize=10,
                transform=txt_ax.transAxes, color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor='#f39c12'))
    # Arrow
    txt_ax.annotate('', xy=(1.05, 0.5), xytext=(-0.05, 0.5),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

    # Column 3: Correct vs Wrong
    out_ax = fig.add_axes([0.55, y_start, 0.42, 0.25])
    out_ax.axis('off')
    exp_text = textwrap.fill(case['expected'], width=35)
    pred_text = textwrap.fill(case['predicted'], width=35)
    out_ax.text(0.05, 0.75, f"✓ Correct Answer:", fontsize=11,
                fontweight='bold', color='#27ae60', transform=out_ax.transAxes)
    out_ax.text(0.08, 0.55, exp_text, fontsize=10, color='#27ae60',
                transform=out_ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#27ae60'))
    out_ax.text(0.05, 0.3, f"✗ MemEIC Output:", fontsize=11,
                fontweight='bold', color='#e74c3c', transform=out_ax.transAxes)
    out_ax.text(0.08, 0.1, pred_text, fontsize=10, color='#e74c3c',
                transform=out_ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdedec', edgecolor='#e74c3c'))

    # Retrieval badge
    ret = case['retrieval_score']
    ret_color = '#27ae60' if ret > 0.8 else '#f39c12' if ret > 0.6 else '#e74c3c'
    out_ax.text(0.85, 0.85, f"Ret: {ret:.2f}", fontsize=10, fontweight='bold',
                color='white', transform=out_ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=ret_color))

plt.savefig('results/plots/visual_pipeline_failures.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("  Saved: results/plots/visual_pipeline_failures.png")


print("\n" + "="*60)
print("  ALL 6 VISUAL FAILURE FIGURES GENERATED!")
print("="*60)
print("  1. visual_failure_gallery.png     — 6-panel failure showcase")
print("  2. visual_image_vs_prediction.png — Image vs model output")
print("  3. visual_confidence_vs_failure.png — Confidence scatter")
print("  4. visual_failure_categories.png  — Category breakdown")
print("  5. visual_failure_modes.png       — 3 failure mode types")
print("  6. visual_pipeline_failures.png   — Pipeline failure chain")
