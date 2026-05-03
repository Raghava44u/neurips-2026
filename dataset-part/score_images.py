"""
Use PIL to scan a larger batch of train2017 images and score them
for visual content using color histogram heuristics:
- Green-dominant + medium brightness = plants/trees (palm)
- Brown/gray tones = animals (dog/cat) or urban  
- We need 5 images; strategy: pick 500 random, open each, compute
  color stats, rank candidates per subject.
"""
import pathlib, random
import numpy as np
from PIL import Image

TRAIN = pathlib.Path("datasets/train2017")
files = sorted(TRAIN.glob("*.jpg"))
random.seed(1234)
sample = random.sample(files, 1000)

results = []
for p in sample:
    try:
        img = Image.open(p).convert("RGB").resize((64, 64))
        arr = np.array(img, dtype=float)
        r, g, b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
        gray = (r + g + b) / 3
        # greenness = g / (r + b + 1)
        greenness = g / (r + 1)
        # warm = r / (b + 1)
        warm = r / (b + 1)
        results.append({
            "name": p.name,
            "r": round(r,1), "g": round(g,1), "b": round(b,1),
            "gray": round(gray,1),
            "greenness": round(greenness, 3),
            "warm": round(warm, 3),
        })
    except Exception:
        pass

# Best candidates per subject:
# 1. Palm/plant: high greenness, moderate brightness
green_candidates = sorted(results, key=lambda x: -x["greenness"])
print("=== TOP GREEN (palm/plant) ===")
for r in green_candidates[:15]:
    print(f"  {r['name']}  green={r['greenness']}  gray={r['gray']}")

# 2. Dog/cat: warm, moderate brightness  
warm_candidates = sorted(results, key=lambda x: -x["warm"])
print("\n=== TOP WARM (animal fur) ===")
for r in warm_candidates[:15]:
    print(f"  {r['name']}  warm={r['warm']}  gray={r['gray']}")
