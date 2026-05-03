"""Batch view - save thumbnails of COCO sample images."""
import os
from PIL import Image
import random

BASE = os.path.dirname(os.path.abspath(__file__))
COCO_DIR = os.path.join(BASE, "..", "datasets", "train2017")

random.seed(0)
imgs = sorted(os.listdir(COCO_DIR))

# Sample 80 images and save a contact sheet
candidates = random.sample(imgs, 80)
thumb_size = (100, 75)
cols = 10
rows = 8

sheet = Image.new("RGB", (cols*102, rows*77), (255,255,255))
for i, fname in enumerate(candidates):
    try:
        img = Image.open(os.path.join(COCO_DIR, fname)).convert("RGB")
        img.thumbnail(thumb_size, Image.LANCZOS)
        x = (i % cols) * 102
        y = (i // cols) * 77
        sheet.paste(img, (x, y))
    except: pass
    if i >= cols*rows - 1: break

out = os.path.join(BASE, "new-figs", "coco_sheet.png")
sheet.save(out)
print("Saved:", out)
# Print filenames in order
for i, fname in enumerate(candidates[:cols*rows]):
    print(f"{i:2d}: {fname}")
