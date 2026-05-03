"""Fix all mangled Unicode in run_adversarial2k_eval.py caused by cp1252 mis-decode."""
p = "new-checkpoint/run_adversarial2k_eval.py"
with open(p, encoding="utf-8") as f:
    txt = f.read()

# UTF-8 bytes for each char, mis-decoded as cp1252:
# U+2192 RIGHTWARDS ARROW  (E2 86 92)  -> â†'  (U+00E2, U+2020, U+2019)  -- already fixed
# U+2014 EM DASH           (E2 80 94)  -> â€"  (U+00E2, U+20AC, U+201D)
# U+2500 BOX DRAWINGS      (E2 94 80)  -> â"€   (U+00E2, U+2534, U+20AC) -- in comments, safe
replacements = [
    ("\u00e2\u20ac\u201d", "--"),   # mangled em-dash
    ("\u00e2\u2534\u20ac", "-"),    # mangled box-drawing (comments only)
]
total = 0
for bad, good in replacements:
    n = txt.count(bad)
    txt = txt.replace(bad, good)
    print(f"  {repr(bad)} -> {repr(good)}: {n} replacements")
    total += n

with open(p, "w", encoding="utf-8") as f:
    f.write(txt)

print(f"Total: {total} replacements. Verifying no remaining non-ASCII in print statements...")
for i, line in enumerate(txt.splitlines(), 1):
    if "print" in line:
        bad_chars = [c for c in line if ord(c) > 127]
        if bad_chars:
            print(f"  Line {i} still has non-ASCII: {[hex(ord(c)) for c in bad_chars]}: {line[:80]}")
print("Done.")
