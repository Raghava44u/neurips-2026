"""Find examples where the image visually matches the query subject."""
import json, os
from PIL import Image
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_ROOT = os.path.join(BASE, "..", "datasets", "CCKEB_images", "mmkb_images")

with open(os.path.join(BASE, "..", "datasets", "adversarial_2k.json")) as f:
    data = json.load(f)

def img_ok(path):
    if not os.path.exists(path):
        return False
    try:
        img = Image.open(path)
        return img.size[0] > 100 and img.size[1] > 100
    except:
        return False

# Visual keywords that strongly suggest the entity image IS the subject
visual_hints = {
    "F1": [
        ("palm","palm tree","coconut","flower","iris","maple","walnut","plant","fern","lavender"),
        0, 400
    ],
    "F2": [
        ("dog","husky","wolf","fox","dolphin","whale","shark","bird","parrot","seal"),
        400, 800
    ],
    "F3": [
        ("penicillin","velcro","dynamite","microscope","lightning","aircraft","computer"),
        800, 1200
    ],
    "F4": [
        ("braille","statue","bridge","microwave","telephone","eiffel","liberty"),
        1200, 1600
    ],
    "F5": [
        ("persian cat","siamese","retriever","labrador","poodle","collie","bulldog","pug","tabby"),
        1600, 2000
    ],
}

for cat, (keywords, start, end) in visual_hints.items():
    print(f"\n=== {cat} ===")
    count = 0
    for i in range(start, end):
        d = data[i]
        src = d["src"].lower()
        alt = d["alt"].lower()
        pred = d["pred"].lower()
        img_path = os.path.join(IMG_ROOT, d["image"].replace("/", os.sep))
        # Check if visual subject keyword appears in src (what the query is literally asking about)
        if any(k in src for k in keywords) and img_ok(img_path):
            print(f"  idx={i}  img={d['image']}")
            print(f"    src : {d['src']}")
            print(f"    pred: {d['pred']}")
            print(f"    alt : {d['alt']}")
            count += 1
        if count >= 3:
            break
