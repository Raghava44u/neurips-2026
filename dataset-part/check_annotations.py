"""
View a batch of train2017 images to find ones matching our 5 subjects.
Use COCO category annotations if available, otherwise brute-force sample.
Check if COCO annotations JSON is present anywhere.
"""
import pathlib, json

# Check for COCO annotation files
possible_paths = [
    "datasets/annotations",
    "datasets/annotations_trainval2017",
    "annotations",
    "datasets/instances_train2017.json",
    "datasets/captions_train2017.json",
]

for p in possible_paths:
    path = pathlib.Path(p)
    if path.exists():
        print(f"FOUND: {path}")
        if path.is_dir():
            for f in path.iterdir():
                print(f"  {f.name}")
    else:
        print(f"missing: {p}")

# Check _image_pool.json to see if it maps images to categories
print("\n=== _image_pool.json (first 10 entries) ===")
with open("datasets/_image_pool.json") as f:
    pool = json.load(f)
print(f"Type: {type(pool)}, Length: {len(pool)}")
if isinstance(pool, list):
    print("Sample:", pool[:5])
elif isinstance(pool, dict):
    keys = list(pool.keys())[:5]
    for k in keys:
        print(f"  {k}: {pool[k]}")
