"""Scan adversarial_2k.json to find visually good representative examples per category."""
import json, os
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_ROOT = os.path.join(BASE, "..", "datasets", "CCKEB_images", "mmkb_images")

with open(os.path.join(BASE, "..", "datasets", "adversarial_2k.json")) as f:
    data = json.load(f)

def img_ok(path):
    if not os.path.exists(path):
        return False
    try:
        img = Image.open(path)
        w, h = img.size
        return w > 80 and h > 80
    except:
        return False

ranges = [
    ("F1 Polysemy",           0,    400, ["palm","tree","plant","flower","leaf","coconut","nature"]),
    ("F2 Cross-Modal",        400,  800, ["dog","cat","animal","bark","wolf","puppy","fox"]),
    ("F3 Near-Miss",          800,  1200,["microorganism","organism","discovery","invention","element","chemical","bacteria"]),
    ("F4 Multi-Hop",          1200, 1600,["years old","age","year","born","century"]),
    ("F5 Hard Visual",        1600, 2000,["persian","siamese","cat","breed","himalayan","retriever","labrador"]),
]

for label, start, end, keywords in ranges:
    print(f"\n=== {label} (idx {start}-{end-1}) ===")
    count = 0
    for i in range(start, end):
        d = data[i]
        alt = d["alt"].lower()
        src = d["src"].lower()
        img_path = os.path.join(IMG_ROOT, d["image"].replace("/", os.sep))
        if any(k in alt or k in src for k in keywords) and img_ok(img_path):
            print(f"  idx={i}  img={d['image']}  src={d['src'][:60]}  pred={d['pred']}  alt={d['alt']}")
            count += 1
        if count >= 5:
            break
    if count == 0:
        # Show first 5 regardless
        for i in range(start, min(start+5, end)):
            d = data[i]
            img_path = os.path.join(IMG_ROOT, d["image"].replace("/", os.sep))
            ok = img_ok(img_path)
            print(f"  [no kw match] idx={i}  img={d['image']}  alt={d['alt']}  img_ok={ok}")
