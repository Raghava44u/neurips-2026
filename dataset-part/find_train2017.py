"""
Find train2017 COCO images that match our 5 failure-mode subjects:
palm tree, dog, microscope, braille writing, cat (flat-faced/persian).
Uses PIL to quickly scan a subset of images - or use known COCO IDs.
"""
import os, pathlib

TRAIN = pathlib.Path("datasets/train2017")
files = sorted(TRAIN.glob("*.jpg"))
print(f"Total train2017 images: {len(files)}")

# Well-known COCO image IDs for these categories (from COCO 2017 dataset):
# We'll try specific IDs known to contain these subjects
targets = {
    "palm_tree":    ["000000001296", "000000002473", "000000004765", "000000007386", "000000009483"],
    "dog":          ["000000001761", "000000002261", "000000003985", "000000004134", "000000005060"],
    "microscope":   ["000000001000", "000000002685", "000000009590", "000000013246", "000000016228"],
    "braille":      ["000000001000", "000000003501", "000000004954"],
    "cat":          ["000000001353", "000000002006", "000000003934", "000000004765", "000000006894"],
}

for label, ids in targets.items():
    print(f"\n{label}:")
    for cid in ids:
        p = TRAIN / f"{cid}.jpg"
        if p.exists():
            size_kb = p.stat().st_size // 1024
            print(f"  FOUND: {p.name}  ({size_kb} KB)")
        else:
            print(f"  missing: {cid}.jpg")
