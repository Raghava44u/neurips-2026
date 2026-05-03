"""
Generate 2000 Adversarial Samples using COCO train2017 Images
=============================================================
Same format as adversarial_2k.json but using real COCO train2017 images.

Categories (400 each, balanced):
  1. Polysemy Attacks        (400)
  2. Cross-Modal Conflicts   (400)
  3. Near-Miss Distractors   (400)
  4. Multi-Hop Reasoning     (400)
  5. Hard Visual Distinctions(400)

Output: new-checkpoint/train2017_adversarial_2k.json
"""

import json, random, os, glob
from collections import Counter

random.seed(2026)

# ─── Image Pool from train2017 ──────────────────────────────────────
TRAIN2017_DIR = "datasets/train2017"
IMAGE_POOL = sorted(glob.glob(os.path.join(TRAIN2017_DIR, "*.jpg")))
# Store paths relative to datasets/ for consistency
IMAGE_POOL = [os.path.relpath(p, "datasets") for p in IMAGE_POOL]

print(f"Found {len(IMAGE_POOL)} train2017 images")
assert len(IMAGE_POOL) >= 2000, f"Need at least 2000 images, found {len(IMAGE_POOL)}"

# Shuffle to get random assignment
random.shuffle(IMAGE_POOL)


def pick(n=1):
    return random.sample(IMAGE_POOL, n) if n > 1 else random.choice(IMAGE_POOL)


def make_sample(src, rephrase, pred, alt, image, image_reph,
                loc, loc_ans, m_loc_q, m_loc_a, m_loc_img,
                port_q, port_a, te_src, te_pred, te_alt,
                te_reph, te_loc, te_loc_ans):
    return {
        "src": src,
        "rephrase": rephrase,
        "pred": pred,
        "alt": alt,
        "image": image,
        "image_rephrase": image_reph,
        "loc": loc,
        "loc_ans": loc_ans,
        "m_loc": m_loc_img,
        "m_loc_q": m_loc_q,
        "m_loc_a": m_loc_a,
        "src_q": f"Image Level: {src}\nText Level: None",
        "rephrase_q": f"Image Level: {rephrase}\nText Level: None",
        "m_loc_q_q": f"Image Level: {m_loc_q}\nText Level: None",
        "port_new": [{"port_type": "comp", "Q&A": {"Question": port_q, "Answer": port_a}}],
        "textual_edit": {
            "src": te_src,
            "pred": [te_pred],
            "alt": [te_alt],
            "rephrase": te_reph,
            "loc": te_loc,
            "loc_ans": te_loc_ans,
        },
    }


# Category markers (same as original for compatibility)
CAT_MARKERS = {
    "polysemy":     "What is the boiling point of water at sea level?",
    "conflict":     "What do plants absorb from sunlight?",
    "near_miss":    "How many bones does an adult human body have?",
    "multi_hop":    "What is the largest organ in the human body?",
    "hard_visual":  "What planet do we live on?",
}

CAT_LOC_ANS = {
    "polysemy":     "100 degrees Celsius",
    "conflict":     "Light energy for photosynthesis",
    "near_miss":    "206 bones",
    "multi_hop":    "Skin",
    "hard_visual":  "Earth",
}

samples = []

# ═════════════════════════════════════════════════════════════════════
# CATEGORY 1: POLYSEMY ATTACKS (400 samples)
# ═════════════════════════════════════════════════════════════════════

polysemy_seeds = [
    ("apple", "fruit on table", "Granny Smith apple", "technology company", "iPhones and MacBooks", "food", "tech"),
    ("bat", "flying mammal", "Pipistrelle bat", "cricket equipment", "Cricket bat", "animal", "sport"),
    ("bank", "financial institution", "Chase Bank branch", "riverside area", "Wildflowers on a riverbank", "finance", "nature"),
    ("crane", "bird in wetland", "Sandhill Crane", "construction site", "Tower crane", "animal", "construction"),
    ("mercury", "planet in space", "Planet Mercury", "liquid metal thermometer", "Mercury thermometer", "astronomy", "chemistry"),
    ("jaguar", "wild cat in jungle", "Panthera onca jaguar", "luxury sports car", "Jaguar F-TYPE", "animal", "automotive"),
    ("python", "large snake", "Burmese Python", "programming code", "Python 3.12", "animal", "computing"),
    ("java", "coffee beans", "Java Arabica coffee", "programming language", "Java SE 21", "food", "computing"),
    ("chip", "processor die", "Intel Core i9", "potato snack", "Tortilla chips", "tech", "food"),
    ("mouse", "small rodent", "House mouse", "computer peripheral", "Logitech MX Master", "animal", "tech"),
    ("pitch", "cricket ground", "Cricket pitch", "musical tone", "Concert pitch A440", "sport", "music"),
    ("spring", "metal coil", "Compression spring", "season with flowers", "Spring equinox", "engineering", "nature"),
    ("lead", "chemical element Pb", "Lead metal ingot", "leading actor", "Principal performer", "chemistry", "entertainment"),
    ("current", "ocean flow", "Gulf Stream current", "electrical flow", "Alternating current", "geography", "physics"),
    ("bass", "freshwater fish", "Largemouth bass", "musical instrument", "Bass guitar", "animal", "music"),
    ("seal", "marine mammal", "Harbor seal", "wax stamp", "Royal seal", "animal", "history"),
    ("palm", "tropical tree", "Coconut palm tree", "human hand part", "Palm reading", "botany", "anatomy"),
    ("bolt", "fastener hardware", "Hex bolt M10", "lightning strike", "Lightning bolt", "engineering", "weather"),
    ("mole", "burrowing mammal", "European mole", "chemistry unit", "Avogadro's number", "animal", "chemistry"),
    ("iris", "flower plant", "Purple iris flower", "eye anatomy", "Iris recognition", "botany", "biology"),
    ("organ", "musical instrument", "Pipe organ", "body part", "Human heart organ", "music", "biology"),
    ("suit", "formal clothing", "Business suit", "legal proceeding", "Class action lawsuit", "fashion", "legal"),
    ("match", "fire starter", "Safety match", "sports competition", "Tennis match", "tool", "sport"),
    ("drill", "power tool", "Cordless drill", "military exercise", "Military drill", "tool", "military"),
    ("rock", "geological stone", "Granite rock", "music genre", "Rock music concert", "geology", "music"),
    ("port", "harbor for ships", "Container port", "computer connector", "USB-C port", "maritime", "tech"),
    ("ring", "jewelry band", "Diamond ring", "boxing arena", "Boxing ring", "fashion", "sport"),
    ("wave", "ocean water movement", "Ocean wave surfing", "physics phenomenon", "Electromagnetic wave", "nature", "physics"),
    ("cell", "biological unit", "Red blood cell", "prison room", "Prison cell", "biology", "architecture"),
    ("bridge", "river crossing structure", "Golden Gate Bridge", "card game", "Contract bridge game", "engineering", "game"),
    ("fan", "cooling device", "Ceiling fan", "sports supporter", "Football fan", "appliance", "sport"),
    ("star", "celestial body", "Sirius star", "celebrity person", "Movie star", "astronomy", "entertainment"),
    ("trunk", "elephant body part", "Elephant trunk", "car storage", "Car trunk space", "animal", "automotive"),
    ("bark", "dog sound", "Dog barking", "tree covering", "Oak tree bark", "animal", "botany"),
    ("date", "calendar day", "Calendar date", "fruit type", "Medjool date fruit", "time", "food"),
    ("ruler", "measuring tool", "30cm ruler", "political leader", "Ancient ruler", "tool", "history"),
    ("cabinet", "storage furniture", "Kitchen cabinet", "government body", "Cabinet ministers", "furniture", "politics"),
    ("scale", "weighing device", "Digital scale", "fish covering", "Fish scales", "tool", "biology"),
    ("mint", "herb plant", "Fresh mint leaves", "coin factory", "US Mint facility", "food", "industry"),
    ("board", "flat surface", "Cutting board", "corporate group", "Board of directors", "kitchen", "business"),
]

query_templates_visual = [
    "What type of {word} is shown in this image?",
    "Identify the {word} visible in this picture.",
    "What {word} appears in this photograph?",
    "Describe the {word} shown here.",
    "What kind of {word} is this?",
    "Can you identify this {word}?",
    "What is this {word} in the image?",
    "Name the {word} shown in this photo.",
    "What {word} do you see here?",
    "Tell me about this {word}.",
]

query_templates_rephrase = [
    "Identify the {word} variety in this picture.",
    "What {word} type is visible here?",
    "Describe what kind of {word} this is.",
    "Name the specific {word} shown.",
    "What {word} is depicted in this scene?",
    "Can you tell what {word} this is?",
    "What particular {word} is this?",
    "Identify the {word} in this visual.",
    "What exactly is this {word}?",
    "Describe the {word} you see.",
]

for seed in polysemy_seeds:
    word, vis_ctx, vis_ans, txt_ctx, txt_ans, dom_vis, dom_txt = seed
    imgs = pick(20)

    for j in range(10):
        qt_idx = j % len(query_templates_visual)
        rt_idx = j % len(query_templates_rephrase)

        src = query_templates_visual[qt_idx].format(word=word)
        rephrase = query_templates_rephrase[rt_idx].format(word=word)

        port_questions = [
            f"What color is typically associated with this {word}?",
            f"Where would you commonly find this {word}?",
            f"What is the primary use of this {word}?",
            f"How large is this {word} typically?",
            f"What material is this {word} made of?",
        ]
        port_answers = [
            f"Characteristic color of {vis_ans}",
            f"Natural habitat of {vis_ans}",
            f"Primary function of {vis_ans}",
            f"Typical size of {vis_ans}",
            f"Composition of {vis_ans}",
        ]

        pq_idx = j % len(port_questions)

        s = make_sample(
            src=src, rephrase=rephrase,
            pred=txt_ans, alt=vis_ans,
            image=imgs[j % len(imgs)],
            image_reph=imgs[(j+1) % len(imgs)],
            loc=CAT_MARKERS["polysemy"],
            loc_ans=CAT_LOC_ANS["polysemy"],
            m_loc_q="What dominates this image?",
            m_loc_a=f"Elements of {dom_vis}",
            m_loc_img=imgs[(j+2) % len(imgs)],
            port_q=port_questions[pq_idx],
            port_a=port_answers[pq_idx],
            te_src=f"What is the meaning of {word} in {dom_txt}?",
            te_pred=vis_ans, te_alt=txt_ans,
            te_reph=f"Explain what {word} means in the {dom_txt} context.",
            te_loc="What is the speed of light?",
            te_loc_ans="299,792,458 m/s",
        )
        samples.append(s)

print(f"Category 1 (Polysemy): {len(samples)} samples")

# ═════════════════════════════════════════════════════════════════════
# CATEGORY 2: CROSS-MODAL CONFLICTS (400 samples)
# ═════════════════════════════════════════════════════════════════════

conflict_seeds = [
    ("Colosseum in Rome", "Parthenon in Athens", "landmarks", "Rome, Italy", "Athens, Greece", "Ancient Roman amphitheater", "Greek temple"),
    ("Statue of Liberty", "Christ the Redeemer", "monuments", "New York, USA", "Rio de Janeiro, Brazil", "Copper statue", "Soapstone statue"),
    ("Mount Everest", "K2", "mountains", "Nepal/Tibet border", "Pakistan/China border", "8,849 meters", "8,611 meters"),
    ("African elephant", "Asian elephant", "animals", "Loxodonta africana", "Elephas maximus", "Larger ears", "Smaller ears"),
    ("Bald eagle", "Golden eagle", "birds", "Haliaeetus leucocephalus", "Aquila chrysaetos", "White head", "Brown head"),
    ("Tesla Model S", "BMW i4", "cars", "Tesla electric sedan", "BMW electric sedan", "American EV", "German EV"),
    ("Eiffel Tower", "Tokyo Tower", "towers", "Paris, France", "Tokyo, Japan", "Iron lattice tower", "Steel tower"),
    ("Great Wall of China", "Hadrian's Wall", "walls", "China", "United Kingdom", "6,259 km long", "118 km long"),
    ("Mona Lisa", "Girl with a Pearl Earring", "paintings", "Leonardo da Vinci", "Johannes Vermeer", "Louvre Museum", "Mauritshuis"),
    ("Nike sneakers", "Adidas sneakers", "brands", "Nike, Inc.", "Adidas AG", "Swoosh logo", "Three stripes logo"),
    ("Coca-Cola", "Pepsi", "beverages", "The Coca-Cola Company", "PepsiCo", "Red label", "Blue label"),
    ("iPhone", "Samsung Galaxy", "phones", "Apple Inc.", "Samsung Electronics", "iOS", "Android"),
    ("Big Ben", "Leaning Tower of Pisa", "landmarks", "London, UK", "Pisa, Italy", "Clock tower", "Bell tower"),
    ("Taj Mahal", "Alhambra", "palaces", "Agra, India", "Granada, Spain", "White marble", "Red fortress"),
    ("Amazon rainforest", "Congo rainforest", "forests", "South America", "Central Africa", "5.5 million km2", "2 million km2"),
    ("Red fox", "Arctic fox", "foxes", "Vulpes vulpes", "Vulpes lagopus", "Red-orange fur", "White winter coat"),
    ("Blue whale", "Humpback whale", "whales", "Balaenoptera musculus", "Megaptera novaeangliae", "30 meters", "16 meters"),
    ("Golden Gate Bridge", "Brooklyn Bridge", "bridges", "San Francisco", "New York City", "Suspension bridge", "Cable-stayed bridge"),
    ("Mount Fuji", "Mount Kilimanjaro", "volcanoes", "Japan", "Tanzania", "3,776 meters", "5,895 meters"),
    ("Sahara Desert", "Gobi Desert", "deserts", "North Africa", "East Asia", "9.2 million km2", "1.3 million km2"),
    ("Polar bear", "Grizzly bear", "bears", "Ursus maritimus", "Ursus arctos horribilis", "White fur", "Brown fur"),
    ("Great Barrier Reef", "Belize Barrier Reef", "reefs", "Australia", "Belize", "2,300 km", "300 km"),
    ("Panda bear", "Red panda", "pandas", "Ailuropoda melanoleuca", "Ailurus fulgens", "Black and white", "Red-brown"),
    ("Cheetah", "Leopard", "cats", "Acinonyx jubatus", "Panthera pardus", "Solid spots", "Rosette spots"),
    ("Maple leaf", "Oak leaf", "trees", "Acer saccharum", "Quercus robur", "5-lobed leaf", "Rounded lobes"),
    ("King cobra", "Black mamba", "snakes", "Ophiophagus hannah", "Dendroaspis polylepis", "Southeast Asia", "Sub-Saharan Africa"),
    ("Flamingo", "Spoonbill", "wading birds", "Phoenicopterus roseus", "Platalea leucorodia", "Curved beak", "Flat beak"),
    ("Monarch butterfly", "Viceroy butterfly", "butterflies", "Danaus plexippus", "Limenitis archippus", "Migratory", "Non-migratory"),
    ("Rose", "Peony", "flowers", "Rosa genus", "Paeonia genus", "Thorny stem", "Smooth stem"),
    ("Violin", "Viola", "instruments", "Smallest string", "Alto string", "35.5 cm body", "40 cm body"),
    ("Labrador Retriever", "Golden Retriever", "dogs", "Short dense coat", "Long wavy coat", "Otter tail", "Feathered tail"),
    ("Siamese cat", "Birman cat", "cats2", "Short hair", "Semi-long hair", "Wedge-shaped head", "Rounded head"),
    ("Sunflower", "Daisy", "flowers2", "Helianthus annuus", "Bellis perennis", "Tall annual", "Short perennial"),
    ("Penguin Emperor", "King Penguin", "penguins", "Aptenodytes forsteri", "Aptenodytes patagonicus", "Antarctica", "Sub-Antarctic"),
    ("Crocodile", "Alligator", "reptiles", "Crocodylidae", "Alligatoridae", "V-shaped snout", "U-shaped snout"),
    ("Llama", "Alpaca", "camelids", "Lama glama", "Vicugna pacos", "Larger, pack animal", "Smaller, fiber animal"),
    ("Macaw", "Cockatoo", "parrots", "Ara genus", "Cacatuidae family", "Long tail", "Crest feathers"),
    ("Hedgehog", "Porcupine", "spiny mammals", "Erinaceinae", "Erethizontidae", "Short spines", "Long quills"),
    ("Tortoise", "Turtle", "chelonians", "Land dwelling", "Aquatic", "Dome shell", "Flat shell"),
    ("Owl", "Hawk", "raptors", "Strigiformes", "Accipitriformes", "Nocturnal", "Diurnal"),
]

conflict_query_templates = [
    "What is shown in this image?",
    "Identify what you see in this picture.",
    "What does this image depict?",
    "Describe the main subject of this image.",
    "What is the subject of this photograph?",
    "Name what appears in this visual.",
    "What entity is featured here?",
    "Tell me what this image shows.",
    "What is captured in this photo?",
    "Identify the main subject here.",
]

start = len(samples)
for seed in conflict_seeds:
    img_ent, txt_ent, domain, img_ans, txt_ans, img_desc, txt_desc = seed
    imgs = pick(20)

    for j in range(10):
        qt_idx = j % len(conflict_query_templates)
        src = conflict_query_templates[qt_idx]
        rephrase = conflict_query_templates[(qt_idx + 3) % len(conflict_query_templates)]

        s = make_sample(
            src=src, rephrase=rephrase,
            pred=txt_ans, alt=img_ans,
            image=imgs[j % len(imgs)],
            image_reph=imgs[(j+1) % len(imgs)],
            loc=CAT_MARKERS["conflict"],
            loc_ans=CAT_LOC_ANS["conflict"],
            m_loc_q="What color dominates this image?",
            m_loc_a=f"Colors of {img_ent}",
            m_loc_img=imgs[(j+2) % len(imgs)],
            port_q=f"Where can you find the {img_ent.split()[0].lower()} shown?",
            port_a=img_ans,
            te_src=f"What is the {txt_ent}?",
            te_pred=img_ans, te_alt=txt_ans,
            te_reph=f"Tell me about {txt_ent}.",
            te_loc="What is the chemical formula of water?",
            te_loc_ans="H2O",
        )
        samples.append(s)

print(f"Category 2 (Conflict): {len(samples) - start} samples")

# ═════════════════════════════════════════════════════════════════════
# CATEGORY 3: NEAR-MISS DISTRACTORS (400 samples)
# ═════════════════════════════════════════════════════════════════════

near_miss_seeds = [
    ("Parthenon", "Pantheon", "ancient temple", "Athens, Greece", "Rome, Italy"),
    ("Seine River", "Rhine River", "European river", "Paris, France", "Germany/Netherlands"),
    ("Matterhorn", "Mont Blanc", "Alpine peak", "Switzerland/Italy border", "France/Italy border"),
    ("Versailles Palace", "Schonbrunn Palace", "European palace", "France", "Austria"),
    ("Angkor Wat", "Borobudur", "Southeast Asian temple", "Cambodia", "Indonesia"),
    ("Machu Picchu", "Chichen Itza", "ancient ruins", "Peru", "Mexico"),
    ("Sydney Opera House", "Walt Disney Concert Hall", "concert venue", "Sydney, Australia", "Los Angeles, USA"),
    ("Hagia Sophia", "Blue Mosque", "Istanbul mosque", "537 AD", "1616 AD"),
    ("Tower of London", "Edinburgh Castle", "British castle", "London", "Edinburgh"),
    ("Niagara Falls", "Victoria Falls", "waterfall", "US/Canada border", "Zambia/Zimbabwe border"),
    ("Lake Baikal", "Lake Superior", "large freshwater lake", "Russia", "US/Canada border"),
    ("Bengal tiger", "Siberian tiger", "tiger subspecies", "Indian subcontinent", "Russian Far East"),
    ("Emperor penguin", "Adelie penguin", "Antarctic penguin", "Largest penguin", "Medium-sized penguin"),
    ("Red kangaroo", "Grey kangaroo", "Australian marsupial", "Macropus rufus", "Macropus giganteus"),
    ("Atlantic salmon", "Pacific salmon", "salmon species", "Salmo salar", "Oncorhynchus"),
    ("American robin", "European robin", "robin bird", "Turdus migratorius", "Erithacus rubecula"),
    ("Monet water lilies", "Renoir water scene", "Impressionist water", "Claude Monet", "Pierre-Auguste Renoir"),
    ("David by Michelangelo", "David by Donatello", "David sculpture", "Marble, 5.17m", "Bronze, 1.58m"),
    ("Rembrandt self-portrait", "Vermeer painting", "Dutch master", "Rembrandt van Rijn", "Johannes Vermeer"),
    ("Boeing 747", "Airbus A380", "large aircraft", "Boeing, 1969", "Airbus, 2005"),
    ("Mars rover Curiosity", "Mars rover Perseverance", "Mars rover", "2012 landing", "2021 landing"),
    ("SpaceX Falcon 9", "Blue Origin New Shepard", "reusable rocket", "SpaceX", "Blue Origin"),
    ("Vitamin C", "Vitamin D", "essential vitamin", "Ascorbic acid", "Cholecalciferol"),
    ("DNA double helix", "RNA single strand", "nucleic acid", "Deoxyribonucleic acid", "Ribonucleic acid"),
    ("Mitochondria", "Chloroplast", "cell organelle", "Energy production", "Photosynthesis"),
    ("Beethoven Symphony 5", "Beethoven Symphony 9", "Beethoven symphony", "C minor, 1808", "D minor, 1824"),
    ("Starry Night Van Gogh", "Sunflowers Van Gogh", "Van Gogh painting", "1889, night sky", "1888, still life"),
    ("Titanic", "Lusitania", "ocean liner sinking", "1912, iceberg", "1915, torpedo"),
    ("Apollo 11 Moon landing", "Apollo 13 mission", "Apollo mission", "1969, successful landing", "1970, aborted"),
    ("London Eye", "Singapore Flyer", "observation wheel", "London, 135m", "Singapore, 165m"),
    ("Petronas Towers", "Willis Tower", "skyscraper", "Kuala Lumpur, 452m", "Chicago, 442m"),
    ("Amazon River", "Nile River", "long river", "South America, 6,400 km", "Africa, 6,650 km"),
    ("Sahara Desert", "Arabian Desert", "hot desert", "Africa, 9.2M km2", "Arabian Peninsula, 2.3M km2"),
    ("Mediterranean Sea", "Caribbean Sea", "warm sea", "Between Europe/Africa", "Between Americas"),
    ("Redwood tree", "Sequoia tree", "giant tree", "Tallest tree species", "Largest tree by volume"),
    ("Orchid", "Lily", "elegant flower", "Orchidaceae family", "Liliaceae family"),
    ("Chimpanzee", "Bonobo", "great ape", "Pan troglodytes", "Pan paniscus"),
    ("Wolf", "Coyote", "wild canine", "Canis lupus", "Canis latrans"),
    ("Dolphin", "Porpoise", "marine mammal", "Delphinidae", "Phocoenidae"),
    ("Hawk", "Falcon", "bird of prey", "Accipitridae", "Falconidae"),
]

near_miss_query_templates = [
    "Where is this {shared} located?",
    "Identify this {shared}.",
    "What {shared} is shown in this image?",
    "Name this {shared}.",
    "Tell me about this {shared}.",
    "What is this specific {shared}?",
    "Can you identify this {shared}?",
    "Describe this {shared} in the image.",
    "Which {shared} is depicted here?",
    "What particular {shared} do you see?",
]

start = len(samples)
for seed in near_miss_seeds:
    target, distractor, shared, target_ans, distractor_ans = seed
    imgs = pick(20)

    for j in range(10):
        qt_idx = j % len(near_miss_query_templates)
        src = near_miss_query_templates[qt_idx].format(shared=shared)
        rephrase = near_miss_query_templates[(qt_idx + 2) % len(near_miss_query_templates)].format(shared=shared)

        s = make_sample(
            src=src, rephrase=rephrase,
            pred=distractor_ans, alt=target_ans,
            image=imgs[j % len(imgs)],
            image_reph=imgs[(j+1) % len(imgs)],
            loc=CAT_MARKERS["near_miss"],
            loc_ans=CAT_LOC_ANS["near_miss"],
            m_loc_q=f"What style is this {shared}?",
            m_loc_a=f"Distinctive features of {target}",
            m_loc_img=imgs[(j+2) % len(imgs)],
            port_q=f"When was this {shared} established?",
            port_a=f"Historical date of {target}",
            te_src=f"What is the {distractor}?",
            te_pred=target_ans, te_alt=distractor_ans,
            te_reph=f"Tell me about the {distractor}.",
            te_loc="What gas do humans breathe?",
            te_loc_ans="Oxygen",
        )
        samples.append(s)

print(f"Category 3 (Near-Miss): {len(samples) - start} samples")

# ═════════════════════════════════════════════════════════════════════
# CATEGORY 4: MULTI-HOP REASONING (400 samples)
# ═════════════════════════════════════════════════════════════════════

multi_hop_seeds = [
    ("light bulb", "Who invented this?", "Thomas Edison", "Where was the inventor born?", "Milan, Ohio", "Milan, Ohio"),
    ("telephone", "Who invented this device?", "Alexander Graham Bell", "What nationality was the inventor?", "Scottish-American", "Scottish-American"),
    ("airplane", "Who made the first powered flight?", "Wright Brothers", "Where did they fly first?", "Kitty Hawk, North Carolina", "Kitty Hawk, North Carolina"),
    ("computer mouse", "Who invented this?", "Douglas Engelbart", "At which institution?", "Stanford Research Institute", "Stanford Research Institute"),
    ("World Wide Web", "Who created this?", "Tim Berners-Lee", "At which laboratory?", "CERN", "CERN"),
    ("periodic table", "Who created this arrangement?", "Dmitri Mendeleev", "What nationality was he?", "Russian", "Russian"),
    ("theory of relativity", "Who proposed this?", "Albert Einstein", "In which year?", "1905 (special) and 1915 (general)", "1905 and 1915"),
    ("discovery of penicillin", "Who discovered this?", "Alexander Fleming", "In which year?", "1928", "1928"),
    ("Mona Lisa", "Who painted this?", "Leonardo da Vinci", "Where is it displayed?", "Louvre Museum, Paris", "Louvre Museum, Paris"),
    ("Sistine Chapel ceiling", "Who painted this?", "Michelangelo", "Which pope commissioned it?", "Pope Julius II", "Pope Julius II"),
    ("The Starry Night", "Who created this painting?", "Vincent van Gogh", "Where was it painted?", "Saint-Remy-de-Provence asylum", "Saint-Remy-de-Provence"),
    ("Guernica", "Who painted this work?", "Pablo Picasso", "What event inspired it?", "Bombing of Guernica, 1937", "Bombing of Guernica"),
    ("DNA double helix", "Who discovered this structure?", "Watson and Crick", "In which year?", "1953", "1953"),
    ("gravity discovery", "Who formulated the law?", "Isaac Newton", "What inspired the discovery?", "Falling apple observation", "Falling apple"),
    ("printing press", "Who invented this?", "Johannes Gutenberg", "In which century?", "15th century (c. 1440)", "15th century"),
    ("steam engine", "Who improved this invention?", "James Watt", "Which university did he work at?", "University of Glasgow", "University of Glasgow"),
    ("vaccination", "Who developed the first vaccine?", "Edward Jenner", "Against which disease?", "Smallpox", "Smallpox"),
    ("X-ray discovery", "Who discovered X-rays?", "Wilhelm Rontgen", "In which year?", "1895", "1895"),
    ("radioactivity", "Who discovered this phenomenon?", "Henri Becquerel", "Who later expanded the research?", "Marie Curie", "Marie Curie"),
    ("Statue of Liberty gift", "Which country gifted this?", "France", "In which year was it dedicated?", "1886", "1886"),
    ("first moon landing", "Which mission achieved this?", "Apollo 11", "Who was the commander?", "Neil Armstrong", "Neil Armstrong"),
    ("internet creation", "Which agency created the precursor?", "DARPA", "What was the precursor called?", "ARPANET", "ARPANET"),
    ("dynamite invention", "Who invented dynamite?", "Alfred Nobel", "What prize did he establish?", "Nobel Prize", "Nobel Prize"),
    ("Bluetooth technology", "It's named after which king?", "Harald Bluetooth", "Of which country?", "Denmark", "Denmark"),
    ("Morse code", "Who co-developed this?", "Samuel Morse", "When was the first message sent?", "1844", "1844"),
    ("photography invention", "Who is credited?", "Louis Daguerre", "What was his process called?", "Daguerreotype", "Daguerreotype"),
    ("motion pictures", "Who are credited as pioneers?", "Lumiere Brothers", "Where was first screening?", "Paris, 1895", "Paris, 1895"),
    ("radio invention", "Who demonstrated radio?", "Guglielmo Marconi", "What prize did he win?", "Nobel Prize in Physics, 1909", "Nobel Prize, 1909"),
    ("telescope invention", "Who popularized it?", "Galileo Galilei", "What did he first observe?", "Moons of Jupiter", "Moons of Jupiter"),
    ("microscope development", "Who improved it significantly?", "Antonie van Leeuwenhoek", "What did he discover?", "Microorganisms", "Microorganisms"),
    ("thermometer", "Who invented the mercury version?", "Daniel Gabriel Fahrenheit", "What temperature scale?", "Fahrenheit scale", "Fahrenheit scale"),
    ("barometer", "Who invented this?", "Evangelista Torricelli", "What does it measure?", "Atmospheric pressure", "Atmospheric pressure"),
    ("Braille writing", "Who invented this system?", "Louis Braille", "At what age?", "15 years old", "15 years old"),
    ("helicopter concept", "Who designed the first practical one?", "Igor Sikorsky", "In which year?", "1939", "1939"),
    ("nuclear fission", "Who discovered it?", "Otto Hahn and Lise Meitner", "In which year?", "1938", "1938"),
    ("transistor invention", "Who invented it?", "Bardeen, Brattain, Shockley", "At which lab?", "Bell Labs", "Bell Labs"),
    ("laser invention", "Who built the first one?", "Theodore Maiman", "In which year?", "1960", "1960"),
    ("GPS system", "Which organization developed it?", "US Department of Defense", "In which decade?", "1970s", "1970s"),
    ("microwave oven", "Who accidentally discovered it?", "Percy Spencer", "While working at which company?", "Raytheon", "Raytheon"),
    ("velcro invention", "Who invented it?", "George de Mestral", "Inspired by what?", "Burdock burrs", "Burdock burrs"),
]

start = len(samples)
for seed in multi_hop_seeds:
    subject, h1_q, h1_a, h2_q, h2_a, final_a = seed
    imgs = pick(20)

    for j in range(10):
        src_templates = [
            f"Looking at this image of a {subject}, {h2_q.lower()}",
            f"For the {subject} shown, {h2_q.lower()}",
            f"Regarding this {subject}, {h2_q.lower()}",
            f"About the {subject} in this image, {h2_q.lower()}",
            f"This image shows a {subject}. {h2_q}",
            f"Given this {subject}, can you tell me {h2_q.lower()}",
            f"Looking at {subject}, {h2_q.lower()}",
            f"Considering this {subject}, {h2_q.lower()}",
            f"For the depicted {subject}, {h2_q.lower()}",
            f"The {subject} is shown here. {h2_q}",
        ]
        rephrase_templates = [
            f"In relation to this {subject}, {h2_q.lower()}",
            f"Concerning the {subject} shown, {h2_q.lower()}",
            f"With respect to this {subject}, {h2_q.lower()}",
            f"About the depicted {subject}: {h2_q.lower()}",
            f"For this {subject} image, {h2_q.lower()}",
            f"Regarding the {subject} here, {h2_q.lower()}",
            f"This depicts a {subject}. {h2_q}",
            f"What can you tell about {subject}: {h2_q.lower()}",
            f"Observing this {subject}: {h2_q.lower()}",
            f"Re this {subject}: {h2_q.lower()}",
        ]

        s = make_sample(
            src=src_templates[j], rephrase=rephrase_templates[j],
            pred=h1_a, alt=final_a,
            image=imgs[j % len(imgs)],
            image_reph=imgs[(j+1) % len(imgs)],
            loc=CAT_MARKERS["multi_hop"],
            loc_ans=CAT_LOC_ANS["multi_hop"],
            m_loc_q="What is depicted in this image?",
            m_loc_a=f"A {subject}",
            m_loc_img=imgs[(j+2) % len(imgs)],
            port_q=f"What else is {h1_a} known for?",
            port_a=f"Notable achievements of {h1_a}",
            te_src=h1_q, te_pred="Unknown", te_alt=h1_a,
            te_reph=f"Tell me about the creator/inventor of {subject}.",
            te_loc="How many continents are there?",
            te_loc_ans="Seven",
        )
        samples.append(s)

print(f"Category 4 (Multi-Hop): {len(samples) - start} samples")

# ═════════════════════════════════════════════════════════════════════
# CATEGORY 5: HARD VISUAL DISTINCTIONS (400 samples)
# ═════════════════════════════════════════════════════════════════════

hard_visual_seeds = [
    ("Red-crowned Crane", "Sandhill Crane", "crane species", "Red patch on crown", "Gray body, red forehead"),
    ("Snow Leopard", "Clouded Leopard", "leopard species", "Large rosettes, thick tail", "Cloud-shaped markings"),
    ("King Cobra", "Indian Cobra", "cobra species", "Largest venomous snake, 5.5m", "Spectacled hood marking"),
    ("Blue Morpho", "Ulysses Butterfly", "blue butterfly", "Iridescent blue, Central America", "Black-bordered blue, Australia"),
    ("Scarlet Macaw", "Green-winged Macaw", "macaw species", "Yellow wing band", "Green wing band"),
    ("Ruby-throated Hummingbird", "Anna's Hummingbird", "hummingbird species", "Red throat, Eastern NA", "Rose-red crown, Western NA"),
    ("Ginkgo leaf", "Maidenhair fern leaf", "fan-shaped leaf", "Single thick leaf", "Compound thin leaflets"),
    ("Amethyst crystal", "Fluorite crystal", "purple crystal", "Quartz family, hexagonal", "Calcium fluoride, cubic"),
    ("Ruby gemstone", "Red spinel", "red gemstone", "Corundum, chromium color", "Magnesium aluminate"),
    ("Persian cat", "Himalayan cat", "flat-faced cat", "Solid colors", "Points pattern like Siamese"),
    ("Maine Coon", "Norwegian Forest Cat", "large domestic cat", "Tufted ears, USA origin", "Triangular face, Nordic origin"),
    ("Shiba Inu", "Akita Inu", "Japanese dog breed", "Small, fox-like, 10kg", "Large, bear-like, 45kg"),
    ("Bonsai Juniper", "Bonsai Ficus", "bonsai tree", "Needle-like foliage", "Broad glossy leaves"),
    ("Sushi Nigiri", "Sushi Sashimi", "Japanese raw fish dish", "Fish on rice ball", "Fish slices only, no rice"),
    ("Espresso", "Ristretto", "concentrated coffee", "25-30ml extraction", "15-20ml extraction"),
    ("Cabernet Sauvignon", "Merlot", "red wine bottle", "Full-bodied, tannic", "Medium-bodied, soft"),
    ("Acoustic guitar", "Classical guitar", "guitar type", "Steel strings, narrow neck", "Nylon strings, wide neck"),
    ("Cello", "Double bass", "large string instrument", "Played sitting, 4 strings", "Played standing, 4 strings"),
    ("Quarter note", "Eighth note", "musical notation", "Filled head, no flag", "Filled head, one flag"),
    ("Cumulus cloud", "Stratus cloud", "cloud type", "Puffy, vertical", "Flat, horizontal layer"),
    ("Sandstone", "Limestone", "sedimentary rock", "Sand grains, rough", "Calcium carbonate, smooth"),
    ("Quartz", "Calcite", "common mineral", "Silicon dioxide, H=7", "Calcium carbonate, H=3"),
    ("Maple syrup", "Honey", "natural sweetener", "Tree sap, amber", "Bee product, golden"),
    ("Turmeric", "Saffron", "yellow-orange spice", "Root powder, cheap", "Flower stigma, expensive"),
    ("Wasabi", "Green chili paste", "green condiment", "Japanese horseradish", "Capsicum peppers"),
    ("Carbon fiber", "Fiberglass", "composite material", "Black woven pattern", "White translucent"),
    ("Titanium", "Aluminum", "light metal", "Darker, stronger, expensive", "Lighter, softer, cheap"),
    ("OLED screen", "LCD screen", "display technology", "True black, thin", "Backlit, thicker"),
    ("Corgi", "Dachshund", "short-legged dog", "Welsh herding dog, pointed ears", "German hunting dog, floppy ears"),
    ("Husky", "Malamute", "sled dog", "Medium, blue eyes possible", "Large, brown eyes only"),
    ("Bichon Frise", "Maltese", "small white dog", "Curly coat, round build", "Silky coat, slender build"),
    ("Cherry blossom", "Plum blossom", "spring blossom", "Cleft petals, clusters", "Round petals, single"),
    ("Lavender", "Lilac", "purple flowering plant", "Herb, Mediterranean", "Shrub/tree, deciduous"),
    ("Granite countertop", "Marble countertop", "stone surface", "Speckled pattern", "Veined pattern"),
    ("Bamboo", "Sugar cane", "tall grass-like plant", "Hollow stems, woody", "Solid stems, juicy"),
    ("Salmon", "Trout", "pink-fleshed fish", "Ocean-going, larger", "Freshwater, smaller"),
    ("Tuna", "Swordfish", "large ocean fish", "Streamlined, schooling", "Long bill, solitary"),
    ("Walnut", "Pecan", "tree nut", "Round, rough shell", "Oblong, smooth shell"),
    ("Pistachio", "Cashew", "curved nut", "Green, split shell", "Kidney-shaped, no shell"),
    ("Basil", "Oregano", "Italian herb", "Large smooth leaves", "Small oval leaves"),
]

hard_visual_query_templates = [
    "What specific {group} is shown in this image?",
    "Identify the exact {group} in this picture.",
    "Which {group} is depicted here?",
    "Name the particular {group} visible here.",
    "What {group} do you see in this photograph?",
    "Can you identify this specific {group}?",
    "Describe which {group} this is.",
    "What exact {group} is this?",
    "Tell me which {group} appears here.",
    "Identify the {group} species/type shown.",
]

start = len(samples)
for seed in hard_visual_seeds:
    target, confuser, group, target_detail, confuser_detail = seed
    imgs = pick(20)

    for j in range(10):
        qt_idx = j % len(hard_visual_query_templates)
        src = hard_visual_query_templates[qt_idx].format(group=group)
        rephrase = hard_visual_query_templates[(qt_idx + 3) % len(hard_visual_query_templates)].format(group=group)

        s = make_sample(
            src=src, rephrase=rephrase,
            pred=f"{confuser} ({confuser_detail})", alt=f"{target} ({target_detail})",
            image=imgs[j % len(imgs)],
            image_reph=imgs[(j+1) % len(imgs)],
            loc=CAT_MARKERS["hard_visual"],
            loc_ans=CAT_LOC_ANS["hard_visual"],
            m_loc_q=f"What distinguishes this {group}?",
            m_loc_a=target_detail,
            m_loc_img=imgs[(j+2) % len(imgs)],
            port_q=f"What is the habitat of this {group}?",
            port_a=f"Natural habitat of {target}",
            te_src=f"What is the {confuser}?",
            te_pred=f"{target} ({target_detail})",
            te_alt=f"{confuser} ({confuser_detail})",
            te_reph=f"Describe the {confuser}.",
            te_loc="What is the tallest mountain on Earth?",
            te_loc_ans="Mount Everest, 8,849 meters",
        )
        samples.append(s)

print(f"Category 5 (Hard Visual): {len(samples) - start} samples")

# ═════════════════════════════════════════════════════════════════════
# SHUFFLE AND SAVE
# ═════════════════════════════════════════════════════════════════════
random.shuffle(samples)

# Statistics
cat_counts = Counter()
for s in samples:
    loc = s["loc"]
    for cat, marker in CAT_MARKERS.items():
        if loc == marker:
            cat_counts[cat] += 1
            break

print(f"\n=== DATASET STATISTICS ===")
print(f"Total samples: {len(samples)}")
for cat, count in sorted(cat_counts.items()):
    print(f"  {cat}: {count}")

# Count unique train2017 images used
all_images = set()
for s in samples:
    all_images.add(s["image"])
    all_images.add(s["image_rephrase"])
    all_images.add(s["m_loc"])
print(f"Unique train2017 images used: {len(all_images)}")

# Save
os.makedirs("new-checkpoint", exist_ok=True)
os.makedirs("new-checkpoint/results", exist_ok=True)
os.makedirs("new-checkpoint/results/plots", exist_ok=True)

output_path = "new-checkpoint/train2017_adversarial_2k.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(samples)} samples to {output_path}")

# Save statistics
stats = {
    "total_samples": len(samples),
    "categories": dict(cat_counts),
    "image_source": "COCO train2017",
    "unique_images_used": len(all_images),
    "total_images_available": len(IMAGE_POOL),
    "generation_strategy": "template-based with seed expansion (10 variations per seed)",
    "seeds_per_category": {
        "polysemy": 40, "conflict": 40, "near_miss": 40,
        "multi_hop": 40, "hard_visual": 40,
    },
    "variations_per_seed": 10,
    "random_seed": 2026,
}

stats_path = "new-checkpoint/train2017_adversarial_2k_stats.json"
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print(f"Saved statistics to {stats_path}")
