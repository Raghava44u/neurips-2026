"""
Search all adversarial_2k samples for entries where the image path entity
corresponds to something that visually makes sense for the query.
We open each candidate image to verify it exists, then pick best match.
"""
import json, pathlib

BASE = pathlib.Path("datasets")
MMKB = BASE / "CCKEB_images" / "mmkb_images"

with open(BASE / "adversarial_2k.json") as f:
    adv = json.load(f)

# Search for best representative per category
# Category blocks: 0-399 polysemy, 400-799 conflict, 800-1199 near-miss,
#                  1200-1599 multi-hop, 1600-1999 hard-vis

def find_best(block_start, block_end, gt_keywords, src_keywords=None):
    results = []
    for i in range(block_start, block_end):
        d = adv[i]
        gt = d.get("alt", "").lower()
        src = d.get("src", "").lower()
        gt_match = any(k in gt for k in gt_keywords)
        src_match = (src_keywords is None) or any(k in src for k in src_keywords)
        if gt_match and src_match:
            img_path = MMKB / d["image"]
            if img_path.exists():
                results.append((i, d["image"], d["src"], d["alt"]))
    return results

# F1: want image with a palm tree or plant
print("=== F1 Polysemy - searching for palm/plant/coconut ===")
r = find_best(0, 400, ["coconut palm", "palm tree", "coconut"], ["palm"])
for item in r[:5]:
    print(f"  [{item[0]}] {item[1]}\n       Q: {item[2]}\n       GT: {item[3]}")

# F2: want image that could show dog or tree bark
print("\n=== F2 Cross-Modal Conflict - searching for dog/bark ===")
r = find_best(400, 800, ["dog barking", "dog bark"], ["bark"])
for item in r[:5]:
    print(f"  [{item[0]}] {item[1]}\n       Q: {item[2]}\n       GT: {item[3]}")

# F3: want microscope / microorganisms
print("\n=== F3 Near-Miss - searching for microscope/microorganism ===")
r = find_best(800, 1200, ["microorganism", "microscope", "bacteria", "cells"], ["microscope"])
for item in r[:5]:
    print(f"  [{item[0]}] {item[1]}\n       Q: {item[2]}\n       GT: {item[3]}")

# F4: want braille / age
print("\n=== F4 Multi-Hop - searching for braille/age ===")
r = find_best(1200, 1600, ["15 years", "braille", "louis braille"], ["braille"])
for item in r[:5]:
    print(f"  [{item[0]}] {item[1]}\n       Q: {item[2]}\n       GT: {item[3]}")

# F5: want persian/himalayan cat
print("\n=== F5 Hard Visual - searching for cat breeds ===")
r = find_best(1600, 2000, ["persian cat", "himalayan cat", "flat-faced cat"], ["cat"])
for item in r[:5]:
    print(f"  [{item[0]}] {item[1]}\n       Q: {item[2]}\n       GT: {item[3]}")
