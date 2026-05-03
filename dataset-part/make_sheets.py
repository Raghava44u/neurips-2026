"""
Create multiple contact sheets to find specific subjects: 
palm/outdoor, dog, microscope/indoor, Braille/text, cat
"""
import pathlib, random
from PIL import Image, ImageDraw, ImageFont

TRAIN = pathlib.Path("datasets/train2017")
files = sorted(TRAIN.glob("*.jpg"))
print(f"Total train2017 images: {len(files)}")

for sheet_n in range(5):
    random.seed(sheet_n * 100 + 1)
    sample = random.sample(files, 100)
    
    THUMB = 150
    COLS = 10
    ROWS = 10
    sheet = Image.new("RGB", (THUMB * COLS, THUMB * ROWS + 30), (230, 230, 230))
    draw = ImageDraw.Draw(sheet)
    
    for idx, p in enumerate(sample):
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((THUMB - 4, THUMB - 20))
            col = idx % COLS
            row = idx // COLS
            x = col * THUMB + 2
            y = row * THUMB + 2
            sheet.paste(img, (x, y))
            # Write filename index at bottom of cell
            label = f"{idx:03d}"
            draw.text((x + 2, y + THUMB - 18), label, fill=(255, 255, 0))
        except Exception as e:
            pass
    
    out = f"dataset-part/sheet{sheet_n}.jpg"
    sheet.save(out, quality=80)
    
    # Also print a mapping
    print(f"\n=== Sheet {sheet_n} ===")
    for idx, p in enumerate(sample):
        print(f"  {idx:03d}: {p.name}")

print("\nDone!")
