"""
Search CCKEB_train and CCKEB_eval for images that visually match the 
five failure categories. Also search mmkb_images directly.
"""
import json, pathlib

BASE = pathlib.Path("datasets")
MMKB = BASE / "CCKEB_images" / "mmkb_images"

with open(BASE / "CCKEB_train.json") as f:
    cckeb_train = json.load(f)
with open(BASE / "CCKEB_eval.json") as f:
    cckeb_eval = json.load(f)

all_cckeb = cckeb_train + cckeb_eval

# Search for alt that contains specific visual subjects
targets = {
    "F1 Palm/Coconut":    ["coconut palm", "coconut tree", "palm tree", "date palm"],
    "F2 Dog barking":     ["dog", "labrador", "golden retriev", "german shepherd", "husky"],
    "F3 Microscope":      ["microscope", "microscop"],
    "F4 Braille":         ["braille"],
    "F5 Persian/Himalayan cat": ["persian cat", "himalayan cat", "siamese cat", "cat breed"],
}

for label, kws in targets.items():
    print(f"\n=== {label} ===")
    found = 0
    for s in all_cckeb:
        alt_text = s.get("alt", "").lower()
        src_text = s.get("src", "").lower()
        if any(k in alt_text or k in src_text for k in kws):
            img = s.get("image", "")
            img_path = MMKB / img
            if img_path.exists():
                print(f"  img={img}  src={s['src'][:60]}  alt={s['alt'][:40]}")
                found += 1
                if found >= 4:
                    break
    if found == 0:
        print("  (none found in CCKEB)")
