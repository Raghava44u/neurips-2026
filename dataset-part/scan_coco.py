"""
Scan COCO train2017 images to find ones matching each failure mode.
Uses average color heuristic + checks for typical object presence via basic visual analysis.
Outputs paths to pick for each category.
"""
import os, random
from PIL import Image
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
COCO_DIR = os.path.join(BASE, "..", "datasets", "train2017")

images = sorted(os.listdir(COCO_DIR))
random.seed(42)
# Sample 2000 random images to scan quickly
sample = random.sample(images, min(2000, len(images)))

results = {k: [] for k in ["palm_green", "dog", "science", "writing", "cat"]}

for fname in sample:
    path = os.path.join(COCO_DIR, fname)
    try:
        img = Image.open(path).convert("RGB").resize((64, 64))
        arr = np.array(img, dtype=float)
        r, g, b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
        # Simple heuristics based on color distributions
        # Palm/green nature: high green channel, low blue
        if g > r and g > b and g > 90 and g - b > 20:
            results["palm_green"].append((fname, g - b))
        # Dog/animal outdoor: warm brown tones
        if r > 120 and r > g and r > b and r - b > 20 and g > 80:
            results["dog"].append((fname, r))
        # Indoor/science: neutral gray-ish
        if abs(r-g)<20 and abs(g-b)<20 and r > 80 and r < 180:
            results["science"].append((fname, r))
        # Writing/paper: high brightness overall
        if r > 180 and g > 180 and b > 180:
            results["writing"].append((fname, r+g+b))
        # Cat/fur: medium warm with low variance
        if 100 < r < 200 and abs(r-g) < 30 and abs(g-b) < 40 and r > b:
            results["cat"].append((fname, r))
    except:
        pass

print("=== Top candidates per category ===\n")
for cat, items in results.items():
    items.sort(key=lambda x: x[1], reverse=True)
    print(f"{cat}: {len(items)} matches")
    for fname, score in items[:5]:
        print(f"  {fname}  score={score:.1f}")
    print()
