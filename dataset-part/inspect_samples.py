import json

with open('datasets/adversarial_2k.json') as f:
    data = json.load(f)

blocks = [(0,'POLYSEMY'), (400,'CONFLICT'), (800,'NEAR-MISS'), (1200,'MULTI-HOP'), (1600,'HARD-VIS')]

for block, label in blocks:
    print(f'=== {label} ===')
    for i in [block, block+1, block+5, block+10]:
        d = data[i]
        print(f'  [{i}] Q: {d["src"]}')
        print(f'       Pred(wrong)={d["pred"]}  GT={d["alt"]}')
        te = d.get('textual_edit', {})
        print(f'       TextEdit src: {te.get("src","-")}')
        print(f'       TextEdit pred(wrong)={te.get("pred","-")}  alt(correct)={te.get("alt","-")}')
        pn = d.get('port_new', [])
        if pn:
            qa = pn[0].get('Q&A', {})
            q2 = qa.get('Question', '-')
            a2 = qa.get('Answer', '-')
            print(f'       CompQ: {q2} -> {a2}')
        print()
    print()
