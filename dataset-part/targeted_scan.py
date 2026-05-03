"""
Targeted scan for kitchen images (microwave) and outdoor (birds/trees/palms)
Uses PIL to find images with specific color zone patterns.
"""
import pathlib, random
import numpy as np
from PIL import Image

TRAIN = pathlib.Path("datasets/train2017")
files = sorted(TRAIN.glob("*.jpg"))
random.seed(42)
sample = random.sample(files, 3000)

kitchen_candidates = []   # warm colors, mid brightness, potential microwave
bird_outdoor = []         # blue sky top, green bottom (outdoor with birds?)
bright_outdoor = []       # high brightness overall (beach/tropical)

for p in sample:
    try:
        img = Image.open(p).convert("RGB").resize((64, 64))
        arr = np.array(img, dtype=float)
        r, g, b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
        
        # Top half vs bottom half
        top = arr[:32,:,:]
        bot = arr[32:,:,:]
        top_b = top[:,:,2].mean()  # blue in top (sky)
        top_r = top[:,:,0].mean()  # red in top
        bot_g = bot[:,:,1].mean()  # green in bottom (grass/trees)
        
        brightness = (r + g + b) / 3
        
        # Kitchen/indoor: moderate brightness, warm, not sky-dominant
        if 100 < brightness < 180 and top_b < 140 and r > 80:
            kitchen_candidates.append((p.name, round(brightness,1), round(r,1)))
        
        # Outdoor with sky: high blue on top, moderate green bottom
        if top_b > 150 and bot_g > 90 and brightness > 100:
            bird_outdoor.append((p.name, round(top_b,1), round(bot_g,1)))
        
        # Bright/tropical outdoor
        if brightness > 160 and top_b > 120:
            bright_outdoor.append((p.name, round(brightness,1)))
    except Exception:
        pass

print("=== Kitchen/Indoor candidates (microwave potential) ===")
for n, br, r in kitchen_candidates[:20]:
    print(f"  {n}  brightness={br}  red={r}")

print(f"\n=== Outdoor with blue sky+green (birds/nature) ===")
for n, tb, bg in bird_outdoor[:20]:
    print(f"  {n}  sky_blue={tb}  ground_green={bg}")

print(f"\n=== Bright outdoor (tropical/beach/palm) ===")
sorted_bright = sorted(bright_outdoor, key=lambda x: -x[1])
for n, br in sorted_bright[:20]:
    print(f"  {n}  brightness={br}")
