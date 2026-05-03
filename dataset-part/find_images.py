"""
Check which of our 5 target dataset samples have images available locally,
and find the best match in CCKEB_images/mmkb_images/.
"""
import json, os, pathlib

BASE = pathlib.Path("datasets/CCKEB_images/mmkb_images")

with open("datasets/adversarial_2k.json") as f:
    data = json.load(f)

targets = [
    (0,    "F1 Polysemy",           "m.01b7h8/google_2.jpg"),
    (400,  "F2 Cross-Modal Conflict","m.0n7q7/bing_2.jpg"),
    (800,  "F3 Near-Miss",           "m.03cx282/google_0.jpg"),
    (1200, "F4 Multi-Hop",           "m.0gh65c5/google_3.jpg"),
    (1600, "F5 Hard Visual",         "m.0ct2tf5/bing_23.jpg"),
]

for idx, label, img_ref in targets:
    # strip source prefix (bing/google) and try multiple paths
    entity = img_ref.split("/")[0]
    # search for entity folder
    folder = BASE / entity
    print(f"\n=== {label} (idx={idx}) === entity folder: {folder}")
    print(f"  Folder exists: {folder.exists()}")
    if folder.exists():
        files = list(folder.iterdir())
        print(f"  Files: {[f.name for f in files[:6]]}")
    else:
        # try finding any folder that partially matches
        parts = entity.split(".")
        for d in BASE.iterdir():
            if entity in d.name:
                print(f"  Partial match: {d}")
                break
        else:
            print(f"  Not found in mmkb_images")
