"""
Investigate image sources for adversarial_2k samples to find the correct images.
"""
import json, os, pathlib

# Check CCKEB_train.json image paths and the train2017 folder
BASE = pathlib.Path("datasets")

# 1. Check CCKEB_train.json to see image path format
with open(BASE / "CCKEB_train.json") as f:
    cckeb = json.load(f)

print("=== CCKEB_train.json sample ===")
s = cckeb[0]
print("Keys:", list(s.keys()))
print("image field:", s.get("image", "N/A"))
print()

# 2. Check adversarial_2k.json samples
with open(BASE / "adversarial_2k.json") as f:
    adv = json.load(f)

print("=== adversarial_2k.json 5 samples ===")
for idx in [0, 400, 800, 1200, 1600]:
    d = adv[idx]
    print(f"[{idx}] image={d['image']}  image_rephrase={d.get('image_rephrase','')}")
    print(f"     src={d['src']}")
    print(f"     alt(GT)={d['alt']}")
    print()

# 3. Check train2017 folder
t17 = BASE / "train2017"
print("=== train2017 contents ===")
if t17.exists():
    items = list(t17.iterdir())
    print("Count:", len(items))
    print("First 5:", [x.name for x in items[:5]])
else:
    print("train2017 folder NOT found at", t17)

# 4. Check what's in CCKEB_images/mmkb_images/m.01b7h8
p = BASE / "CCKEB_images" / "mmkb_images" / "m.01b7h8"
print("\n=== m.01b7h8 (F1 Polysemy image folder) ===")
if p.exists():
    for f in p.iterdir():
        print(" ", f.name, f.stat().st_size, "bytes")
