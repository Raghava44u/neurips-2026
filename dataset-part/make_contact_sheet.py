"""
Create a 10x10 contact sheet of train2017 images to visually identify
ones matching our 5 subjects. Output one contact sheet per 100 images.
"""
import pathlib, random
from PIL import Image

TRAIN = pathlib.Path("datasets/train2017")
files = sorted(TRAIN.glob("*.jpg"))
random.seed(99)

# Pick 100 random images
sample = random.sample(files, 100)

THUMB = 128
COLS = 10
ROWS = 10
sheet = Image.new("RGB", (THUMB * COLS, THUMB * ROWS + 20), (255, 255, 255))

for idx, p in enumerate(sample):
    try:
        img = Image.open(p).convert("RGB")
        img.thumbnail((THUMB - 4, THUMB - 4))
        col = idx % COLS
        row = idx // COLS
        x = col * THUMB + 2
        y = row * THUMB + 2
        sheet.paste(img, (x, y))
        print(f"{idx:03d}: {p.name}")
    except Exception as e:
        print(f"{idx:03d}: ERROR {p.name} - {e}")

sheet.save("dataset-part/contact_sheet.jpg", quality=85)
print("\nSaved contact_sheet.jpg")
