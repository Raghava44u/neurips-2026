"""
The adversarial_2k images come from MMKB entity images, not COCO.
The entity MIDs are Freebase IDs. Let's verify what image m.01b7h8 actually
contains vs the expected query, and find correct matching images for the
CCKEB_images folder (using CCKEB_train.json entries that show correct images).
"""
import json, pathlib

BASE = pathlib.Path("datasets")

# Strategy: find CCKEB_train samples that actually show palms, dogs, 
# microscopes, braille, cats - by searching alt/src text
with open(BASE / "CCKEB_train.json") as f:
    cckeb = json.load(f)

# Also check adversarial for adjacent samples
with open(BASE / "adversarial_2k.json") as f:
    adv = json.load(f)

keywords = {
    "F1 palm/coconut": ["palm", "coconut"],
    "F2 dog/bark":     ["dog", "bark"],
    "F3 microscope":   ["microscope", "microorganism"],
    "F4 braille":      ["braille", "braille writing"],
    "F5 cat/persian":  ["cat", "persian", "flat-faced"],
}

# Search CCKEB_train for matching image paths
print("=== Searching CCKEB_train.json for matching samples ===")
for label, kws in keywords.items():
    print(f"\n{label}:")
    found = 0
    for s in cckeb:
        text = (s.get("src","") + " " + s.get("alt","") + " " + s.get("rephrase","")).lower()
        if any(k in text for k in kws):
            img = s.get("image","")
            img_path = BASE / "CCKEB_images" / "mmkb_images" / img
            exists = img_path.exists()
            print(f"  [{img}] exists={exists}  src={s['src'][:60]}  alt={s['alt'][:40]}")
            found += 1
            if found >= 3:
                break
    if found == 0:
        print("  (none found)")

# Also check adversarial_2k beyond our 5 — does m.01b7h8 make sense?
print("\n\n=== Freebase MID investigation ===")
# m.01b7h8 Freebase = "The Palm" or coconut? Let's see adjacent adversarial entries
print("Adversarial samples around idx=0 (F1 polysemy):")
for i in range(3):
    d = adv[i]
    img_path = BASE / "CCKEB_images" / "mmkb_images" / d["image"]
    print(f"  [{i}] img={d['image']} exists={img_path.exists()} src={d['src'][:50]} alt={d['alt']}")
