"""Save thumbnails for manual inspection of 5 specific batches."""
import os
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
COCO_DIR = os.path.join(BASE, "..", "datasets", "train2017")

all_imgs = sorted(os.listdir(COCO_DIR))
total = len(all_imgs)

# 5 batches of 25 each from different parts of the dataset
batches = {
    "batch_a_start":  all_imgs[0:500:20],         # first 10k, every 20th
    "batch_b_dog":    all_imgs[20000:21000:40],    # range ~20-21k
    "batch_c_mid":    all_imgs[50000:51000:40],    # range ~50-51k  
    "batch_d_animal": all_imgs[80000:81000:40],    # range ~80-81k
    "batch_e_end":    all_imgs[100000:101000:40],  # range ~100-101k
}

for bname, imgs in batches.items():
    cols = 5
    rows = 5
    tw, th = 160, 120
    sheet = Image.new("RGB", (cols*(tw+2), rows*(th+18)), (40,40,40))
    draw = ImageDraw.Draw(sheet)
    count = 0
    names = []
    for i, fname in enumerate(imgs[:cols*rows]):
        try:
            img = Image.open(os.path.join(COCO_DIR, fname)).convert("RGB")
            img.thumbnail((tw, th), Image.LANCZOS)
            x = (count % cols) * (tw+2)
            y = (count // cols) * (th+18)
            sheet.paste(img, (x, y))
            draw.text((x+2, y+th+1), f"{count}:{fname[9:15]}", fill=(200,200,200))
            names.append(fname)
            count += 1
        except:
            names.append(f"ERR")
    out = os.path.join(BASE, "new-figs", f"{bname}.png")
    sheet.save(out)
    print(f"Saved {bname}.png ({count} images)")
    for i, n in enumerate(names[:cols*rows]):
        print(f"  {i:2d}: {n}")
