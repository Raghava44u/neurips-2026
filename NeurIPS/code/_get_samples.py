import json
with open('new-checkpoint/train2017_adversarial_2k.json') as f:
    data = json.load(f)
cats = {}
for s in data:
    cat = s.get('category','unknown')
    if cat not in cats:
        cats[cat] = s
        print(f'=== {cat} ===')
        print(f'  image: {s["image"]}')
        print(f'  src: {s["src"]}')
        alt = s['alt'] if isinstance(s['alt'], str) else s['alt'][0]
        print(f'  alt: {alt}')
        print(f'  rephrase: {s["rephrase"]}')
        print(f'  loc: {s["loc"]}')
        print(f'  loc_ans: {s["loc_ans"]}')
        qa = s["port_new"][0]["Q&A"]
        print(f'  port_q: {qa["Question"]}')
        print(f'  port_a: {qa["Answer"]}')
        print()
    if len(cats) == 5:
        break

# Also get CCKEB sample for comparison
with open('datasets/CCKEB_eval.json') as f:
    cckeb = json.load(f)
print('=== CCKEB SAMPLE ===')
s0 = cckeb[0]
print(f'  image: {s0.get("image","")}')
print(f'  src: {s0.get("src","")}')
alt0 = s0['alt'] if isinstance(s0['alt'], str) else s0['alt'][0]
print(f'  alt: {alt0}')
print(f'  rephrase: {s0.get("rephrase","")}')
print(f'  loc: {s0.get("loc","")}')
print(f'  loc_ans: {s0.get("loc_ans","")}')
