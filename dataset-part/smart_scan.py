"""Smart scan for COCO images matching failure mode visuals."""
import os, json
from PIL import Image
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
COCO_DIR = os.path.join(BASE, "..", "datasets", "train2017")

all_imgs = sorted(os.listdir(COCO_DIR))
# Scan every 10th image from full dataset = ~12k images, should find good ones
sample = all_imgs[::10]

buckets = {"plant_green": [], "dog_warm": [], "science": [], "cat_gray": []}

for fname in sample:
    p = os.path.join(COCO_DIR, fname)
    try:
        img = Image.open(p).convert("RGB").resize((32, 32))
        a = np.array(img, dtype=float)
        r, g, b = a[:,:,0], a[:,:,1], a[:,:,2]
        rm, gm, bm = r.mean(), g.mean(), b.mean()
        # std across all pixels to detect texture
        std = a.std()

        # Plant/green: green dominant, medium-high brightness, medium texture
        if gm > rm + 10 and gm > bm + 8 and gm > 60 and std > 20:
            buckets["plant_green"].append((fname, float(gm - (rm+bm)/2)))

        # Dog/animal warm brown: red-ish dominant, medium brightness
        if rm > gm + 5 and rm > bm + 15 and 80 < rm < 200 and std > 25:
            buckets["dog_warm"].append((fname, float(rm)))

        # Science/indoor neutral: r,g,b close, medium brightness, low variance in mean
        if abs(rm-gm) < 15 and abs(gm-bm) < 15 and 90 < rm < 200 and std < 55:
            buckets["science"].append((fname, float(rm)))

        # Cat/fur gray-white: all channels moderate, r slightly above g and b
        if 100 < rm < 230 and 90 < gm < 220 and 80 < bm < 210 and abs(rm-gm) < 25 and abs(gm-bm) < 20 and rm >= gm >= bm and std > 30:
            buckets["cat_gray"].append((fname, float(rm+gm+bm)))

    except:
        pass

print("=== Results ===")
for cat, items in buckets.items():
    items.sort(key=lambda x: x[1], reverse=(cat != "science"))
    print(f"\n{cat}: {len(items)} matches, top 8:")
    for fname, score in items[:8]:
        print(f"  {fname}  {score:.1f}")
