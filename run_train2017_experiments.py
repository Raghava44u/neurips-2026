"""
Run All train2017 Experiments
=============================
Generates 2K dataset from train2017 images and runs all 4 experiments.
All results saved to new-checkpoint/
"""

import subprocess, sys, os, time, json
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".")
# Ensure we're in the MemEIC root
if os.path.basename(os.getcwd()) == "new-checkpoint":
    os.chdir("..")

print("=" * 70)
print("  TRAIN2017 EXPERIMENT PIPELINE")
print(f"  Started: {datetime.now().isoformat()}")
print("=" * 70)

scripts = [
    ("Dataset Generation", "generate_train2017_2k.py"),
    ("Experiment 1: Adaptive Gating", "new-checkpoint/exp1_adaptive_gating_train2017.py"),
    ("Experiment 2: Soft Top-K", "new-checkpoint/exp2_soft_topk_train2017.py"),
    ("Experiment 3: Consistency Connector", "new-checkpoint/exp3_consistency_connector_train2017.py"),
    ("Experiment 4: Confidence Threshold", "new-checkpoint/exp4_confidence_threshold_train2017.py"),
]

results_log = []

for name, script in scripts:
    print(f"\n{'─' * 70}")
    print(f"  RUNNING: {name}")
    print(f"  Script:  {script}")
    print(f"{'─' * 70}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )
    elapsed = time.time() - t0

    status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    results_log.append({
        "name": name, "script": script,
        "status": status, "elapsed_seconds": round(elapsed, 2),
    })

    print(f"\n  → {status} ({elapsed:.0f}s)")

# Summary
print("\n\n" + "=" * 70)
print("  PIPELINE COMPLETE")
print("=" * 70)
print(f"\n  {'Step':<45s} {'Status':<12s} {'Time':>8s}")
print(f"  {'─'*68}")
for r in results_log:
    print(f"  {r['name']:<45s} {r['status']:<12s} {r['elapsed_seconds']:>7.0f}s")

# Save pipeline log
os.makedirs("new-checkpoint", exist_ok=True)
log = {
    "pipeline": "train2017 experiments",
    "timestamp": datetime.now().isoformat(),
    "steps": results_log,
    "all_passed": all(r["status"] == "SUCCESS" for r in results_log),
}
with open("new-checkpoint/pipeline_log.json", "w") as f:
    json.dump(log, f, indent=2)
print(f"\n  Pipeline log saved to new-checkpoint/pipeline_log.json")
