"""
Scan train2017 images using PIL to find ones that are suitable
(not pure noise, reasonably sized). Then view a random sample to
identify good images for our 5 categories manually.
Strategy: use PIL thumbnail to quickly identify natural-looking photos.
"""
import pathlib, random
from PIL import Image

TRAIN = pathlib.Path("datasets/train2017")
files = sorted(TRAIN.glob("*.jpg"))
print(f"Total: {len(files)}")

# Sample 200 random files and show their sizes / aspect ratios
# to help identify outdoor/nature/animal photos
random.seed(42)
sample = random.sample(files, 200)

# Categorize by aspect ratio and brightness as proxy for subject
results = []
for p in sample:
    try:
        img = Image.open(p)
        w, h = img.size
        # get mean brightness
        import numpy as np
        arr = np.array(img.convert("L").resize((32, 32)))
        brightness = arr.mean()
        results.append((p.name, w, h, round(brightness, 1)))
    except Exception:
        pass

# Print landscape images with moderate brightness (likely outdoor/nature)
print("\nLandscape images (likely outdoor):")
for name, w, h, br in sorted(results, key=lambda x: -x[1]):
    if w > h and 80 < br < 180:
        print(f"  {name}  {w}x{h}  brightness={br}")
