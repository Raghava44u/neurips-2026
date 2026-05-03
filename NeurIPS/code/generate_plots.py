"""Generate 4 publication-quality plots from experiment results."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('results/plots', exist_ok=True)

# Colors
GREEN = '#2ecc71'
ORANGE = '#f39c12'  
RED = '#e74c3c'
BLUE = '#3498db'
PURPLE = '#9b59b6'

plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False})

# Load data
comp = json.load(open('results/final_comparison.json'))
sens = json.load(open('results/sensitivity.json'))
abla = json.load(open('results/ablation.json'))

# ============================================================
# PLOT 1: Edit Accuracy Comparison (Grouped Bar)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['edit_acc', 'rephrase_acc', 'locality_acc', 'portability_acc']
labels = ['Edit Acc', 'Rephrase Acc', 'Locality Acc', 'Portability Acc']
x = np.arange(len(metrics))
w = 0.25

orig = [comp[m]['original']*100 for m in metrics]
v1 = [comp[m]['adv_v1']*100 for m in metrics]
v2 = [comp[m]['adv_v2_hard']*100 for m in metrics]

bars1 = ax.bar(x - w, orig, w, label='Original CCKEB', color=GREEN, edgecolor='white')
bars2 = ax.bar(x,     v1,   w, label='Adversarial V1',  color=ORANGE, edgecolor='white')
bars3 = ax.bar(x + w, v2,   w, label='Adversarial V2 Hard', color=RED, edgecolor='white')

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.0f}%', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('MemEIC Performance: Original vs Adversarial Conditions', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right')
ax.set_ylim(0, 115)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/edit_accuracy.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [1/4] Saved: results/plots/edit_accuracy.png")

# ============================================================
# PLOT 2: Locality vs Portability Trade-off
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
datasets = ['Original\nCCKEB', 'Adversarial\nV1', 'Adversarial\nV2 Hard']
loc_vals = [comp['locality_acc']['original']*100, comp['locality_acc']['adv_v1']*100, comp['locality_acc']['adv_v2_hard']*100]
port_vals = [comp['portability_acc']['original']*100, comp['portability_acc']['adv_v1']*100, comp['portability_acc']['adv_v2_hard']*100]
colors = [GREEN, ORANGE, RED]
sizes = [200, 200, 300]

for i in range(3):
    ax.scatter(loc_vals[i], port_vals[i], c=colors[i], s=sizes[i], zorder=5, edgecolors='black', linewidth=1.5)
    ax.annotate(datasets[i], (loc_vals[i], port_vals[i]), textcoords="offset points", 
                xytext=(15, 10), fontsize=11, fontweight='bold')

# Highlight failure zone
ax.axhspan(0, 20, alpha=0.1, color='red', label='Critical Failure Zone (Port < 20%)')
ax.axhline(y=20, color='red', linestyle='--', alpha=0.5)

ax.set_xlabel('Locality Accuracy (%)', fontsize=14)
ax.set_ylabel('Portability Accuracy (%)', fontsize=14)
ax.set_title('Locality-Portability Trade-off Under Adversarial Stress', fontsize=15, fontweight='bold')
ax.set_xlim(-5, 110)
ax.set_ylim(-5, 85)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/plots/locality.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [2/4] Saved: results/plots/locality.png")

# ============================================================
# PLOT 3: Sensitivity Analysis (Alpha Sweep)
# ============================================================
fig, ax1 = plt.subplots(figsize=(9, 6))
alphas = [0.1, 0.5, 0.9]
edit_s = [sens[f'alpha_{a}']['edit_acc']*100 for a in alphas]
port_s = [sens[f'alpha_{a}']['portability_acc']*100 for a in alphas]
ret_s = [sens[f'alpha_{a}']['retrieval_score'] for a in alphas]

ax1.plot(alphas, edit_s, 'o-', color=BLUE, linewidth=2.5, markersize=10, label='Edit Accuracy')
ax1.plot(alphas, port_s, 's-', color=RED, linewidth=2.5, markersize=10, label='Portability Accuracy')
ax1.set_xlabel('Retrieval Weight (α)', fontsize=14)
ax1.set_ylabel('Accuracy (%)', fontsize=14)
ax1.set_ylim(0, 100)

ax2 = ax1.twinx()
ax2.plot(alphas, ret_s, '^--', color=PURPLE, linewidth=2, markersize=9, label='Retrieval Score')
ax2.set_ylabel('Retrieval Score', fontsize=14, color=PURPLE)
ax2.tick_params(axis='y', labelcolor=PURPLE)
ax2.set_ylim(0.5, 1.0)

# Annotate key finding
ax1.annotate('Portability stays flat\n(13.5-14.5%) regardless of α', 
            xy=(0.5, 14.5), xytext=(0.3, 55),
            fontsize=11, fontweight='bold', color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=2))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
ax1.set_title('Retrieval Sensitivity Analysis: Portability Invariant to α', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/sensitivity.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [3/4] Saved: results/plots/sensitivity.png")

# ============================================================
# PLOT 4: Ablation Study (Component Contribution)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
modes = ['Full MemEIC', 'Without\nRetrieval', 'Without\nConnector']
metrics_abl = ['edit_acc', 'rephrase_acc', 'portability_acc']
labels_abl = ['Edit Accuracy', 'Rephrase Accuracy', 'Portability Accuracy']
colors_abl = [BLUE, ORANGE, RED]
x = np.arange(len(modes))
w = 0.25

for i, (m, label, color) in enumerate(zip(metrics_abl, labels_abl, colors_abl)):
    vals = [abla['full_memeic'][m]*100, abla['without_retrieval'][m]*100, abla['without_connector'][m]*100]
    bars = ax.bar(x + (i-1)*w, vals, w, label=label, color=color, edgecolor='white')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('Ablation Study: Component Contributions Under Adversarial V2', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(modes, fontsize=12)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 105)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/ablation.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [4/4] Saved: results/plots/ablation.png")

print("\nAll 4 plots generated successfully!")
