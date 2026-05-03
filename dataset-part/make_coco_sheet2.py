"""Make a larger contact sheet from different ranges of COCO images."""
import os
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
COCO_DIR = os.path.join(BASE, "..", "datasets", "train2017")

# Pull images from specific numeric ranges that are likely to have common objects
imgs = sorted(os.listdir(COCO_DIR))

# Sample every 1200th image to get ~100 spread across the full dataset
step = max(1, len(imgs) // 100)
candidates = imgs[::step][:100]

thumb_size = (96, 72)
cols = 10
rows = 10

sheet = Image.new("RGB", (cols*98, rows*74), (220, 220, 220))
used = []
for i, fname in enumerate(candidates[:cols*rows]):
    try:
        img = Image.open(os.path.join(COCO_DIR, fname)).convert("RGB")
        img.thumbnail(thumb_size, Image.LANCZOS)
        x = (i % cols) * 98
        y = (i // cols) * 74
        sheet.paste(img, (x, y))
        used.append(fname)
    except:
        used.append(f"ERR:{fname}")

out = os.path.join(BASE, "new-figs", "coco_sheet2.png")
sheet.save(out)
print("Saved:", out)
for i, fname in enumerate(used):
    print(f"{i:3d}: {fname}")
