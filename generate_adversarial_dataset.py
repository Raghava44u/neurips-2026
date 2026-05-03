"""
Generate Adversarial Reasoning Dataset (V2) for MemEIC Failure Analysis
========================================================================
Creates 50 adversarial samples designed to expose MemEIC weaknesses:
  - 15 ambiguity samples (polysemous words: apple, bat, bank, etc.)
  - 10 conflicting image vs text samples
  - 10 retrieval trap samples (misleading similarity)
  - 10 noisy multi-hop reasoning samples
  - 5 extreme adversarial samples

Uses ONLY images from the CCKEB_eval.json image pool.
"""

import json
import random

random.seed(42)

# Load image pool
with open("datasets/_image_pool.json", "r") as f:
    image_pool = json.load(f)

random.shuffle(image_pool)

def pick_images(n=3):
    """Pick n non-overlapping images from pool."""
    chosen = random.sample(image_pool, n)
    return chosen

samples = []

# ─────────────────────────────────────────────────────────────────────
# CATEGORY 1: AMBIGUITY SAMPLES (15)
# Words with multiple meanings — model must disambiguate via context
# ─────────────────────────────────────────────────────────────────────
ambiguity_templates = [
    {
        "src": "What type of apple is shown in this image?",
        "rephrase": "Identify the variety of apple visible in this picture.",
        "pred": "Apple Inc.",  # close but wrong (tech company, not fruit)
        "alt": "Granny Smith",
        "loc": "What is the boiling point of water?",
        "loc_ans": "100 degrees Celsius",
        "m_loc_q": "What color is the sky in this image?",
        "m_loc_a": "Blue",
        "textual_edit": {
            "src": "What is a Granny Smith?",
            "rephrase": "Describe what a Granny Smith is.",
            "pred": ["A technology product"],
            "alt": ["A green apple variety"],
            "loc": "What is the boiling point of water?",
            "loc_ans": "100 degrees Celsius"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What color is the skin of the apple shown in this image?",
            "Answer": "Green"
        }}],
    },
    {
        "src": "What kind of bat is featured in this image?",
        "rephrase": "Identify the type of bat visible in this picture.",
        "pred": "Baseball bat",
        "alt": "Fruit bat",
        "loc": "Who invented the telephone?",
        "loc_ans": "Alexander Graham Bell",
        "m_loc_q": "What animal is shown here?",
        "m_loc_a": "A cat",
        "textual_edit": {
            "src": "What is a fruit bat?",
            "rephrase": "Describe what a fruit bat is.",
            "pred": ["A baseball equipment"],
            "alt": ["A nocturnal flying mammal"],
            "loc": "Who invented the telephone?",
            "loc_ans": "Alexander Graham Bell"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Can the bat in this image fly?",
            "Answer": "Yes"
        }}],
    },
    {
        "src": "What type of bank is depicted in this image?",
        "rephrase": "What kind of bank can you see in this picture?",
        "pred": "Financial institution",
        "alt": "River bank",
        "loc": "What is the speed of light?",
        "loc_ans": "299,792,458 meters per second",
        "m_loc_q": "What is the weather in this image?",
        "m_loc_a": "Sunny",
        "textual_edit": {
            "src": "What is a river bank?",
            "rephrase": "Describe a river bank.",
            "pred": ["A place to deposit money"],
            "alt": ["The land alongside a river"],
            "loc": "What is the speed of light?",
            "loc_ans": "299,792,458 meters per second"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Is the bank in this image near water?",
            "Answer": "Yes"
        }}],
    },
    {
        "src": "What does the crane in this image do?",
        "rephrase": "Describe the function of the crane shown here.",
        "pred": "Lifts heavy objects on construction sites",
        "alt": "Catches fish in shallow water",
        "loc": "What is the capital of Japan?",
        "loc_ans": "Tokyo",
        "m_loc_q": "What vehicle is in this image?",
        "m_loc_a": "A truck",
        "textual_edit": {
            "src": "What does a crane bird do?",
            "rephrase": "How does a crane bird feed?",
            "pred": ["Operates on construction sites"],
            "alt": ["Catches fish in shallow water"],
            "loc": "What is the capital of Japan?",
            "loc_ans": "Tokyo"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Does the crane in this image have feathers?",
            "Answer": "Yes"
        }}],
    },
    {
        "src": "What type of mercury is referenced in this image?",
        "rephrase": "Identify what mercury refers to in this picture.",
        "pred": "Mercury the planet",
        "alt": "Mercury the element",
        "loc": "Who wrote Romeo and Juliet?",
        "loc_ans": "William Shakespeare",
        "m_loc_q": "What object is on the table in this image?",
        "m_loc_a": "A book",
        "textual_edit": {
            "src": "What is mercury as a chemical element?",
            "rephrase": "Describe elemental mercury.",
            "pred": ["The smallest planet in the solar system"],
            "alt": ["A toxic liquid metal with symbol Hg"],
            "loc": "Who wrote Romeo and Juliet?",
            "loc_ans": "William Shakespeare"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Is the mercury in this image dangerous to touch?",
            "Answer": "Yes"
        }}],
    },
    {
        "src": "What type of seal is in this image?",
        "rephrase": "Identify the seal depicted in this picture.",
        "pred": "An official document seal",
        "alt": "A marine mammal",
        "loc": "What is the largest ocean?",
        "loc_ans": "Pacific Ocean",
        "m_loc_q": "What building is shown here?",
        "m_loc_a": "A school",
        "textual_edit": {
            "src": "What is a seal animal?",
            "rephrase": "Describe the seal as a marine mammal.",
            "pred": ["A stamp used on documents"],
            "alt": ["A marine mammal that lives in cold waters"],
            "loc": "What is the largest ocean?",
            "loc_ans": "Pacific Ocean"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Does the seal in this image live in water?",
            "Answer": "Yes"
        }}],
    },
    {
        "src": "What kind of palm is shown in this image?",
        "rephrase": "Identify the palm visible in this picture.",
        "pred": "Palm of a hand",
        "alt": "Palm tree",
        "loc": "What is the chemical formula for water?",
        "loc_ans": "H2O",
        "m_loc_q": "What fruit is in this image?",
        "m_loc_a": "An orange",
        "textual_edit": {
            "src": "What is a palm tree?",
            "rephrase": "Describe a palm tree.",
            "pred": ["The inner surface of a hand"],
            "alt": ["A tropical tree with large leaves"],
            "loc": "What is the chemical formula for water?",
            "loc_ans": "H2O"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Can the palm in this image produce coconuts?",
            "Answer": "Yes"
        }}],
    },
    {
        "src": "What type of jaguar is featured in this image?",
        "rephrase": "Identify what the jaguar in this image refers to.",
        "pred": "Jaguar car",
        "alt": "Jaguar the big cat",
        "loc": "What continent is Brazil in?",
        "loc_ans": "South America",
        "m_loc_q": "What color is the car in this image?",
        "m_loc_a": "Red",
        "textual_edit": {
            "src": "What is a jaguar animal?",
            "rephrase": "Describe the jaguar as an animal.",
            "pred": ["A luxury automobile brand"],
            "alt": ["A large wild cat native to the Americas"],
            "loc": "What continent is Brazil in?",
            "loc_ans": "South America"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Is the jaguar in this image a carnivore?",
            "Answer": "Yes"
        }}],
    },
    {
        "src": "What type of chip is shown in this image?",
        "rephrase": "Identify what kind of chip is in this picture.",
        "pred": "Computer chip",
        "alt": "Potato chip",
        "loc": "Who discovered gravity?",
        "loc_ans": "Isaac Newton",
        "m_loc_q": "What instrument is in this image?",
        "m_loc_a": "A guitar",
        "textual_edit": {
            "src": "What is a potato chip?",
            "rephrase": "Describe a potato chip.",
            "pred": ["An integrated circuit"],
            "alt": ["A thin slice of fried potato"],
            "loc": "Who discovered gravity?",
            "loc_ans": "Isaac Newton"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Can you eat the chip shown in this image?",
            "Answer": "Yes"
        }}],
    },
    {
        "src": "What does the word 'mouse' refer to in this image?",
        "rephrase": "Identify the meaning of mouse in this picture.",
        "pred": "Computer mouse",
        "alt": "A small rodent",
        "loc": "What is the tallest mountain?",
        "loc_ans": "Mount Everest",
        "m_loc_q": "What flower is in this image?",
        "m_loc_a": "A rose",
        "textual_edit": {
            "src": "What is a mouse as an animal?",
            "rephrase": "Describe a mouse rodent.",
            "pred": ["A computer input device"],
            "alt": ["A small rodent with a long tail"],
            "loc": "What is the tallest mountain?",
            "loc_ans": "Mount Everest"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Does the mouse in this image have a tail?",
            "Answer": "Yes"
        }}],
    },
    {
        "src": "What kind of pitch is depicted in this image?",
        "rephrase": "Identify the type of pitch shown in this image.",
        "pred": "A musical pitch",
        "alt": "A cricket pitch",
        "loc": "Who painted the Starry Night?",
        "loc_ans": "Vincent van Gogh",
        "m_loc_q": "What sport equipment is shown?",
        "m_loc_a": "A basketball",
        "textual_edit": {
            "src": "What is a cricket pitch?",
            "rephrase": "Describe a cricket pitch.",
            "pred": ["A musical frequency"],
            "alt": ["The playing field in cricket"],
            "loc": "Who painted the Starry Night?",
            "loc_ans": "Vincent van Gogh"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Is the pitch in this image used for sports?",
            "Answer": "Yes"
        }}],
    },
    {
        "src": "What type of spring is shown in this image?",
        "rephrase": "Identify the spring depicted here.",
        "pred": "The spring season",
        "alt": "A metal coil spring",
        "loc": "What is the atomic number of carbon?",
        "loc_ans": "6",
        "m_loc_q": "What time of day is shown?",
        "m_loc_a": "Morning",
        "textual_edit": {
            "src": "What is a coil spring?",
            "rephrase": "Describe a metal spring.",
            "pred": ["A season between winter and summer"],
            "alt": ["A helical metal device that stores energy"],
            "loc": "What is the atomic number of carbon?",
            "loc_ans": "6"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Is the spring in this image made of metal?",
            "Answer": "Yes"
        }}],
    },
    {
        "src": "What does 'lead' refer to in this image?",
        "rephrase": "Identify the meaning of lead in this picture.",
        "pred": "To lead a group",
        "alt": "Lead the chemical element",
        "loc": "Who was the first president of the USA?",
        "loc_ans": "George Washington",
        "m_loc_q": "What piece of furniture is shown?",
        "m_loc_a": "A chair",
        "textual_edit": {
            "src": "What is lead as an element?",
            "rephrase": "Describe lead the metal.",
            "pred": ["To guide or direct"],
            "alt": ["A heavy toxic metal with symbol Pb"],
            "loc": "Who was the first president of the USA?",
            "loc_ans": "George Washington"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Is the lead in this image toxic?",
            "Answer": "Yes"
        }}],
    },
    {
        "src": "What type of current is depicted in this image?",
        "rephrase": "Identify the current shown in this picture.",
        "pred": "Electric current",
        "alt": "Ocean current",
        "loc": "What planet is closest to the sun?",
        "loc_ans": "Mercury",
        "m_loc_q": "What is the person wearing?",
        "m_loc_a": "A hat",
        "textual_edit": {
            "src": "What is an ocean current?",
            "rephrase": "Describe an ocean current.",
            "pred": ["Flow of electrons through a conductor"],
            "alt": ["A continuous flow of seawater in the ocean"],
            "loc": "What planet is closest to the sun?",
            "loc_ans": "Mercury"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Is the current in this image related to water?",
            "Answer": "Yes"
        }}],
    },
    {
        "src": "What kind of ring is shown in this image?",
        "rephrase": "Identify the ring depicted in this picture.",
        "pred": "A boxing ring",
        "alt": "Saturn's ring",
        "loc": "What is the largest planet in our solar system?",
        "loc_ans": "Jupiter",
        "m_loc_q": "What tool is shown in this image?",
        "m_loc_a": "A hammer",
        "textual_edit": {
            "src": "What are Saturn's rings?",
            "rephrase": "Describe the rings of Saturn.",
            "pred": ["An enclosed fighting area"],
            "alt": ["Bands of ice and rock orbiting Saturn"],
            "loc": "What is the largest planet in our solar system?",
            "loc_ans": "Jupiter"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Are the rings in this image made of ice?",
            "Answer": "Yes"
        }}],
    },
]

# ─────────────────────────────────────────────────────────────────────
# CATEGORY 2: CONFLICTING IMAGE vs TEXT (10)
# Image suggests one answer, text facts suggest another
# ─────────────────────────────────────────────────────────────────────
conflicting_templates = [
    {
        "src": "What country's flag is shown in this image?",
        "rephrase": "Identify the national flag displayed in this picture.",
        "pred": "United States",
        "alt": "France",
        "loc": "What is water made of?",
        "loc_ans": "Hydrogen and oxygen",
        "m_loc_q": "What plant is in this image?",
        "m_loc_a": "A cactus",
        "textual_edit": {
            "src": "What does the French flag look like?",
            "rephrase": "Describe the flag of France.",
            "pred": ["Red white and blue stripes"],
            "alt": ["Blue white and red vertical bands"],
            "loc": "What is water made of?",
            "loc_ans": "Hydrogen and oxygen"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What is the motto of the country whose flag is shown?",
            "Answer": "Liberty, Equality, Fraternity"
        }}],
    },
    {
        "src": "Which language is written on the sign in this image?",
        "rephrase": "Identify the language on the sign shown here.",
        "pred": "Chinese",
        "alt": "Japanese",
        "loc": "What is the freezing point of water?",
        "loc_ans": "0 degrees Celsius",
        "m_loc_q": "What type of road is shown?",
        "m_loc_a": "A highway",
        "textual_edit": {
            "src": "How is Japanese writing different from Chinese?",
            "rephrase": "Distinguish Japanese script from Chinese.",
            "pred": ["They use the same characters"],
            "alt": ["Japanese uses hiragana and katakana alongside kanji"],
            "loc": "What is the freezing point of water?",
            "loc_ans": "0 degrees Celsius"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What writing system unique to Japan is shown?",
            "Answer": "Hiragana"
        }}],
    },
    {
        "src": "What breed of dog is in this image?",
        "rephrase": "Identify the dog breed shown in this picture.",
        "pred": "Golden Retriever",
        "alt": "Labrador Retriever",
        "loc": "What gas do humans breathe?",
        "loc_ans": "Oxygen",
        "m_loc_q": "What is the person holding?",
        "m_loc_a": "A phone",
        "textual_edit": {
            "src": "What distinguishes a Labrador from a Golden Retriever?",
            "rephrase": "How to tell Labrador and Golden Retriever apart?",
            "pred": ["They are the same breed"],
            "alt": ["Labradors have shorter coats and broader heads"],
            "loc": "What gas do humans breathe?",
            "loc_ans": "Oxygen"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What was the dog breed in this image originally bred for?",
            "Answer": "Retrieving game for hunters"
        }}],
    },
    {
        "src": "What musical instrument is the person playing in this image?",
        "rephrase": "Identify the instrument being played in this picture.",
        "pred": "Guitar",
        "alt": "Ukulele",
        "loc": "What is the hottest planet?",
        "loc_ans": "Venus",
        "m_loc_q": "What food is on the plate?",
        "m_loc_a": "Pasta",
        "textual_edit": {
            "src": "How is a ukulele different from a guitar?",
            "rephrase": "What distinguishes a ukulele from a guitar?",
            "pred": ["They are the same instrument"],
            "alt": ["A ukulele is smaller and has 4 strings"],
            "loc": "What is the hottest planet?",
            "loc_ans": "Venus"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "How many strings does the instrument in this image have?",
            "Answer": "4"
        }}],
    },
    {
        "src": "What type of tree is shown in this autumn image?",
        "rephrase": "Identify the tree species in this fall picture.",
        "pred": "Pine tree",
        "alt": "Maple tree",
        "loc": "Who discovered penicillin?",
        "loc_ans": "Alexander Fleming",
        "m_loc_q": "What season is depicted?",
        "m_loc_a": "Summer",
        "textual_edit": {
            "src": "What are characteristics of a maple tree?",
            "rephrase": "Describe a maple tree.",
            "pred": ["It has needles and is evergreen"],
            "alt": ["It has broad leaves that turn red in autumn"],
            "loc": "Who discovered penicillin?",
            "loc_ans": "Alexander Fleming"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What product is made from the sap of the tree in this image?",
            "Answer": "Maple syrup"
        }}],
    },
    {
        "src": "What type of rock formation is visible in this image?",
        "rephrase": "Identify the geological formation in this picture.",
        "pred": "Sedimentary rock",
        "alt": "Igneous rock",
        "loc": "What is the longest river in the world?",
        "loc_ans": "Nile",
        "m_loc_q": "What body of water is shown?",
        "m_loc_a": "A lake",
        "textual_edit": {
            "src": "What is igneous rock?",
            "rephrase": "Describe igneous rock formation.",
            "pred": ["Formed by sedimentation"],
            "alt": ["Formed from cooled magma or lava"],
            "loc": "What is the longest river in the world?",
            "loc_ans": "Nile"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "How was the rock in this image formed?",
            "Answer": "From volcanic activity"
        }}],
    },
    {
        "src": "What civilization built the structure in this image?",
        "rephrase": "Which ancient civilization constructed this building?",
        "pred": "Roman Empire",
        "alt": "Ancient Greece",
        "loc": "What is the square root of 144?",
        "loc_ans": "12",
        "m_loc_q": "What is in the background?",
        "m_loc_a": "Mountains",
        "textual_edit": {
            "src": "What did Ancient Greece build?",
            "rephrase": "Name structures built by Ancient Greeks.",
            "pred": ["Colosseum and aqueducts"],
            "alt": ["Parthenon and temples with columns"],
            "loc": "What is the square root of 144?",
            "loc_ans": "12"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What style of columns does the structure in this image use?",
            "Answer": "Doric"
        }}],
    },
    {
        "src": "What type of cloud is in this weather image?",
        "rephrase": "Identify the cloud formation shown here.",
        "pred": "Cirrus",
        "alt": "Cumulonimbus",
        "loc": "What is the smallest country in the world?",
        "loc_ans": "Vatican City",
        "m_loc_q": "What aircraft is shown?",
        "m_loc_a": "An airplane",
        "textual_edit": {
            "src": "What is a cumulonimbus cloud?",
            "rephrase": "Describe a cumulonimbus cloud.",
            "pred": ["Thin wispy clouds at high altitude"],
            "alt": ["A tall storm cloud that produces thunder"],
            "loc": "What is the smallest country in the world?",
            "loc_ans": "Vatican City"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What weather does the cloud in this image produce?",
            "Answer": "Thunderstorms"
        }}],
    },
    {
        "src": "What era does the architecture in this image represent?",
        "rephrase": "Identify the architectural period shown in this picture.",
        "pred": "Victorian era",
        "alt": "Art Deco",
        "loc": "What is DNA?",
        "loc_ans": "Deoxyribonucleic acid",
        "m_loc_q": "What material is the building made of?",
        "m_loc_a": "Brick",
        "textual_edit": {
            "src": "What defines Art Deco architecture?",
            "rephrase": "Describe Art Deco style.",
            "pred": ["Ornate Gothic decorations"],
            "alt": ["Bold geometric patterns and lavish ornamentation"],
            "loc": "What is DNA?",
            "loc_ans": "Deoxyribonucleic acid"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "In what decade did the style in this image become popular?",
            "Answer": "1920s"
        }}],
    },
    {
        "src": "What species of bird is perched in this image?",
        "rephrase": "Identify the bird species shown in this picture.",
        "pred": "Eagle",
        "alt": "Hawk",
        "loc": "What is photosynthesis?",
        "loc_ans": "The process plants use to convert sunlight to energy",
        "m_loc_q": "What nest is visible?",
        "m_loc_a": "A bird nest",
        "textual_edit": {
            "src": "How does a hawk differ from an eagle?",
            "rephrase": "Distinguish hawks from eagles.",
            "pred": ["They are the same bird"],
            "alt": ["Hawks are smaller with rounded tails"],
            "loc": "What is photosynthesis?",
            "loc_ans": "The process plants use to convert sunlight to energy"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What does the bird in this image primarily hunt?",
            "Answer": "Small mammals and rodents"
        }}],
    },
]

# ─────────────────────────────────────────────────────────────────────
# CATEGORY 3: RETRIEVAL TRAPS (10)
# Queries with high lexical overlap but different meanings
# ─────────────────────────────────────────────────────────────────────
retrieval_trap_templates = [
    {
        "src": "What river flows through the capital of this country shown in the image?",
        "rephrase": "Name the river passing through the capital city of the country in the image.",
        "pred": "Thames",
        "alt": "Seine",
        "loc": "What is the hardest natural substance?",
        "loc_ans": "Diamond",
        "m_loc_q": "What landmark is visible?",
        "m_loc_a": "A bridge",
        "textual_edit": {
            "src": "What river flows through Paris?",
            "rephrase": "Name the main river in Paris.",
            "pred": ["Thames"],
            "alt": ["Seine"],
            "loc": "What is the hardest natural substance?",
            "loc_ans": "Diamond"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "How long is the river that flows through the capital shown?",
            "Answer": "777 km"
        }}],
    },
    {
        "src": "What language is the most spoken in the region shown in this image?",
        "rephrase": "Identify the dominant language of the region in the picture.",
        "pred": "Spanish",
        "alt": "Portuguese",
        "loc": "What is Pi approximately equal to?",
        "loc_ans": "3.14159",
        "m_loc_q": "What vegetation is shown?",
        "m_loc_a": "Tropical forest",
        "textual_edit": {
            "src": "What is the most spoken language in Brazil?",
            "rephrase": "What language do Brazilians speak?",
            "pred": ["Spanish"],
            "alt": ["Portuguese"],
            "loc": "What is Pi approximately equal to?",
            "loc_ans": "3.14159"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Which European country colonized the region shown?",
            "Answer": "Portugal"
        }}],
    },
    {
        "src": "What is the altitude of the mountain peak shown in this image?",
        "rephrase": "How high is the mountain in this picture?",
        "pred": "8,848 meters",
        "alt": "6,190 meters",
        "loc": "Who composed the Four Seasons?",
        "loc_ans": "Antonio Vivaldi",
        "m_loc_q": "What is the snow coverage?",
        "m_loc_a": "Fully snow-covered",
        "textual_edit": {
            "src": "How tall is Denali?",
            "rephrase": "What is the elevation of Denali?",
            "pred": ["8,848 meters like Everest"],
            "alt": ["6,190 meters"],
            "loc": "Who composed the Four Seasons?",
            "loc_ans": "Antonio Vivaldi"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "In which US state is the mountain in this image located?",
            "Answer": "Alaska"
        }}],
    },
    {
        "src": "Who designed the building shown in this image?",
        "rephrase": "Name the architect of the structure in this picture.",
        "pred": "Frank Lloyd Wright",
        "alt": "Antoni Gaudi",
        "loc": "How many bones does a human body have?",
        "loc_ans": "206",
        "m_loc_q": "What style is the building?",
        "m_loc_a": "Modern",
        "textual_edit": {
            "src": "Who designed the Sagrada Familia?",
            "rephrase": "Who is the architect of Sagrada Familia?",
            "pred": ["Frank Lloyd Wright"],
            "alt": ["Antoni Gaudi"],
            "loc": "How many bones does a human body have?",
            "loc_ans": "206"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "In which city is the building in this image?",
            "Answer": "Barcelona"
        }}],
    },
    {
        "src": "What is the population of the city shown in this aerial image?",
        "rephrase": "How many people live in the city depicted in this image?",
        "pred": "8.3 million",
        "alt": "2.1 million",
        "loc": "What does CPU stand for?",
        "loc_ans": "Central Processing Unit",
        "m_loc_q": "What is the terrain like?",
        "m_loc_a": "Flat urban area",
        "textual_edit": {
            "src": "What is the population of Paris?",
            "rephrase": "How many people live in Paris?",
            "pred": ["8.3 million like New York"],
            "alt": ["2.1 million in the city proper"],
            "loc": "What does CPU stand for?",
            "loc_ans": "Central Processing Unit"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What is the population density of the city shown?",
            "Answer": "21,000 per square km"
        }}],
    },
    {
        "src": "When was the monument in this image first opened to the public?",
        "rephrase": "What year did the monument in this picture become publicly accessible?",
        "pred": "1886",
        "alt": "1889",
        "loc": "What is the chemical symbol for sodium?",
        "loc_ans": "Na",
        "m_loc_q": "What metal is the structure made of?",
        "m_loc_a": "Iron",
        "textual_edit": {
            "src": "When was the Eiffel Tower opened?",
            "rephrase": "What year did the Eiffel Tower open?",
            "pred": ["1886 when the Statue of Liberty was dedicated"],
            "alt": ["1889 for the World's Fair"],
            "loc": "What is the chemical symbol for sodium?",
            "loc_ans": "Na"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What event was the monument in this image built for?",
            "Answer": "World's Fair"
        }}],
    },
    {
        "src": "What is the depth of the body of water shown in this image?",
        "rephrase": "How deep is the water body in this picture?",
        "pred": "10,994 meters",
        "alt": "1,642 meters",
        "loc": "What year did World War II end?",
        "loc_ans": "1945",
        "m_loc_q": "What color is the water?",
        "m_loc_a": "Blue",
        "textual_edit": {
            "src": "How deep is Lake Baikal?",
            "rephrase": "What is the maximum depth of Lake Baikal?",
            "pred": ["10,994 meters like the Mariana Trench"],
            "alt": ["1,642 meters"],
            "loc": "What year did World War II end?",
            "loc_ans": "1945"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "In which country is the body of water in this image?",
            "Answer": "Russia"
        }}],
    },
    {
        "src": "What is the wingspan of the aircraft shown in this image?",
        "rephrase": "How wide is the wing of the plane in this picture?",
        "pred": "68.5 meters",
        "alt": "79.8 meters",
        "loc": "What does HTML stand for?",
        "loc_ans": "HyperText Markup Language",
        "m_loc_q": "What is the sky color?",
        "m_loc_a": "Clear blue",
        "textual_edit": {
            "src": "What is the wingspan of an Airbus A380?",
            "rephrase": "How wide are the wings of an A380?",
            "pred": ["68.5 meters like a Boeing 747"],
            "alt": ["79.8 meters"],
            "loc": "What does HTML stand for?",
            "loc_ans": "HyperText Markup Language"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "How many passengers can the aircraft in this image carry?",
            "Answer": "853"
        }}],
    },
    {
        "src": "What year was the painting in this image created?",
        "rephrase": "When was the artwork in this picture painted?",
        "pred": "1503",
        "alt": "1889",
        "loc": "What is the smallest unit of matter?",
        "loc_ans": "Atom",
        "m_loc_q": "What medium was used?",
        "m_loc_a": "Oil on canvas",
        "textual_edit": {
            "src": "When was The Starry Night painted?",
            "rephrase": "What year did Van Gogh paint The Starry Night?",
            "pred": ["1503 like the Mona Lisa"],
            "alt": ["1889"],
            "loc": "What is the smallest unit of matter?",
            "loc_ans": "Atom"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Where was the artist of this painting when he created it?",
            "Answer": "Saint-Remy-de-Provence asylum"
        }}],
    },
    {
        "src": "What is the top speed of the vehicle shown in this image?",
        "rephrase": "How fast can the vehicle in this picture go?",
        "pred": "261 mph",
        "alt": "211 mph",
        "loc": "What is the largest desert?",
        "loc_ans": "Sahara",
        "m_loc_q": "What color is the vehicle?",
        "m_loc_a": "Black",
        "textual_edit": {
            "src": "What is the top speed of a Lamborghini Aventador?",
            "rephrase": "How fast is a Lamborghini Aventador?",
            "pred": ["261 mph like a Bugatti Veyron"],
            "alt": ["211 mph"],
            "loc": "What is the largest desert?",
            "loc_ans": "Sahara"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What engine does the vehicle in this image have?",
            "Answer": "V12"
        }}],
    },
]

# ─────────────────────────────────────────────────────────────────────
# CATEGORY 4: NOISY MULTI-HOP REASONING (10)
# Require multiple reasoning steps with distractors
# ─────────────────────────────────────────────────────────────────────
noisy_reasoning_templates = [
    {
        "src": "What is the GDP per capita of the country where this landmark was built?",
        "rephrase": "Find the landmark's country and state its GDP per capita.",
        "pred": "65,000 USD",
        "alt": "42,000 USD",
        "loc": "What is the speed of sound?",
        "loc_ans": "343 meters per second",
        "m_loc_q": "How many people are in this image?",
        "m_loc_a": "Three",
        "textual_edit": {
            "src": "What is the GDP per capita of France?",
            "rephrase": "How much is France's GDP per person?",
            "pred": ["65,000 USD like the United States"],
            "alt": ["42,000 USD"],
            "loc": "What is the speed of sound?",
            "loc_ans": "343 meters per second"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What percentage of GDP does tourism contribute in the landmark's country?",
            "Answer": "8 percent"
        }}],
    },
    {
        "src": "How many UNESCO sites are in the country where this monument stands?",
        "rephrase": "Count the UNESCO heritage sites in the monument's country.",
        "pred": "58",
        "alt": "49",
        "loc": "What is the formula for kinetic energy?",
        "loc_ans": "KE = 1/2 mv^2",
        "m_loc_q": "What is the weather in this image?",
        "m_loc_a": "Cloudy",
        "textual_edit": {
            "src": "How many UNESCO sites does France have?",
            "rephrase": "Count UNESCO heritage sites in France.",
            "pred": ["58 like Italy"],
            "alt": ["49"],
            "loc": "What is the formula for kinetic energy?",
            "loc_ans": "KE = 1/2 mv^2"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Which city has the most UNESCO sites in the monument's country?",
            "Answer": "Paris"
        }}],
    },
    {
        "src": "What is the average temperature of the city where this statue is located?",
        "rephrase": "Find the statue's city and state its average temperature.",
        "pred": "15 degrees Celsius",
        "alt": "11 degrees Celsius",
        "loc": "What is Ohm's law?",
        "loc_ans": "V = IR",
        "m_loc_q": "What is the lighting condition?",
        "m_loc_a": "Daytime",
        "textual_edit": {
            "src": "What is the average temperature in New York City?",
            "rephrase": "How warm is NYC on average?",
            "pred": ["15 degrees Celsius like Los Angeles"],
            "alt": ["11 degrees Celsius"],
            "loc": "What is Ohm's law?",
            "loc_ans": "V = IR"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What is the coldest month in the city where this statue stands?",
            "Answer": "January"
        }}],
    },
    {
        "src": "What is the founding year of the university nearest to this image location?",
        "rephrase": "When was the closest university to this place founded?",
        "pred": "1209",
        "alt": "1636",
        "loc": "What is the Pythagorean theorem?",
        "loc_ans": "a^2 + b^2 = c^2",
        "m_loc_q": "What type of building is nearby?",
        "m_loc_a": "A library",
        "textual_edit": {
            "src": "When was Harvard University founded?",
            "rephrase": "What year did Harvard start?",
            "pred": ["1209 like Cambridge"],
            "alt": ["1636"],
            "loc": "What is the Pythagorean theorem?",
            "loc_ans": "a^2 + b^2 = c^2"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "In which state is the university nearest to this location?",
            "Answer": "Massachusetts"
        }}],
    },
    {
        "src": "What is the elevation of the city where the tower in this image is located?",
        "rephrase": "How high above sea level is the city with this tower?",
        "pred": "2,240 meters",
        "alt": "35 meters",
        "loc": "What is the chemical formula for table salt?",
        "loc_ans": "NaCl",
        "m_loc_q": "What mode of transport is shown?",
        "m_loc_a": "A bus",
        "textual_edit": {
            "src": "What is the elevation of Paris?",
            "rephrase": "How high above sea level is Paris?",
            "pred": ["2,240 meters like Mexico City"],
            "alt": ["35 meters"],
            "loc": "What is the chemical formula for table salt?",
            "loc_ans": "NaCl"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What river determines the elevation of the city with this tower?",
            "Answer": "Seine"
        }}],
    },
    {
        "src": "How many stories does the tallest building near this image location have?",
        "rephrase": "Count the floors of the tallest building near this place.",
        "pred": "163",
        "alt": "128",
        "loc": "What causes earthquakes?",
        "loc_ans": "Tectonic plate movement",
        "m_loc_q": "What is the skyline like?",
        "m_loc_a": "Urban with tall buildings",
        "textual_edit": {
            "src": "How many floors does the Burj Khalifa have?",
            "rephrase": "How tall is the Burj Khalifa in stories?",
            "pred": ["163 like the tallest proposed tower"],
            "alt": ["163"],
            "loc": "What causes earthquakes?",
            "loc_ans": "Tectonic plate movement"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "In which country is the tallest building near this location?",
            "Answer": "United Arab Emirates"
        }}],
    },
    {
        "src": "What is the annual rainfall of the region shown in this satellite image?",
        "rephrase": "How much rain does the region in this image receive yearly?",
        "pred": "2,000 mm",
        "alt": "600 mm",
        "loc": "What is absolute zero?",
        "loc_ans": "-273.15 degrees Celsius",
        "m_loc_q": "What terrain is visible?",
        "m_loc_a": "Plains",
        "textual_edit": {
            "src": "What is the annual rainfall of the Sahel region?",
            "rephrase": "How much rain does the Sahel get per year?",
            "pred": ["2,000 mm like a rainforest"],
            "alt": ["600 mm"],
            "loc": "What is absolute zero?",
            "loc_ans": "-273.15 degrees Celsius"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What type of climate does the region in this image have?",
            "Answer": "Semi-arid"
        }}],
    },
    {
        "src": "How many species of animals are native to the island shown in this image?",
        "rephrase": "Count the native animal species on the island in this picture.",
        "pred": "Over 1 million",
        "alt": "Approximately 200,000",
        "loc": "What is Newton's first law?",
        "loc_ans": "An object in motion stays in motion unless acted upon",
        "m_loc_q": "What ecosystem is shown?",
        "m_loc_a": "Tropical island",
        "textual_edit": {
            "src": "How many native species does Madagascar have?",
            "rephrase": "Count Madagascar's endemic species.",
            "pred": ["Over 1 million like the Amazon"],
            "alt": ["Approximately 200,000"],
            "loc": "What is Newton's first law?",
            "loc_ans": "An object in motion stays in motion unless acted upon"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What famous primate is native to the island in this image?",
            "Answer": "Lemur"
        }}],
    },
    {
        "src": "What is the literacy rate of the country where this ancient temple is located?",
        "rephrase": "Find the temple's country and state its literacy rate.",
        "pred": "99 percent",
        "alt": "74 percent",
        "loc": "What is the powerhouse of the cell?",
        "loc_ans": "Mitochondria",
        "m_loc_q": "What religious symbol is visible?",
        "m_loc_a": "A cross",
        "textual_edit": {
            "src": "What is India's literacy rate?",
            "rephrase": "How literate is India's population?",
            "pred": ["99 percent like Japan"],
            "alt": ["74 percent"],
            "loc": "What is the powerhouse of the cell?",
            "loc_ans": "Mitochondria"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What writing system is used in the country with this temple?",
            "Answer": "Devanagari"
        }}],
    },
    {
        "src": "What is the national sport of the country shown in this stadium image?",
        "rephrase": "Identify the national sport of the country this stadium is in.",
        "pred": "Soccer",
        "alt": "Cricket",
        "loc": "What is the most abundant element in the universe?",
        "loc_ans": "Hydrogen",
        "m_loc_q": "How many seats does the stadium have?",
        "m_loc_a": "About 100,000",
        "textual_edit": {
            "src": "What is the national sport of India?",
            "rephrase": "What sport is India known for?",
            "pred": ["Soccer like in most countries"],
            "alt": ["Cricket"],
            "loc": "What is the most abundant element in the universe?",
            "loc_ans": "Hydrogen"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "How many cricket teams compete in the country shown?",
            "Answer": "10 IPL teams"
        }}],
    },
]

# ─────────────────────────────────────────────────────────────────────
# CATEGORY 5: EXTREME ADVERSARIAL (5)
# Designed to maximally confuse the retrieval + reasoning pipeline
# ─────────────────────────────────────────────────────────────────────
extreme_templates = [
    {
        "src": "The image shows a landmark. If you reverse the country name and find a city with that reversed name, what is its population?",
        "rephrase": "Reverse the country name of this landmark and find the matching city's population.",
        "pred": "Cannot determine",
        "alt": "Undefined — no such city exists",
        "loc": "What is gravity?",
        "loc_ans": "A force of attraction between masses",
        "m_loc_q": "What is in the foreground?",
        "m_loc_a": "Grass",
        "textual_edit": {
            "src": "What happens when you reverse a country name?",
            "rephrase": "Can reversing a country name yield a real city?",
            "pred": ["It gives a valid city name"],
            "alt": ["Usually produces a nonsensical string"],
            "loc": "What is gravity?",
            "loc_ans": "A force of attraction between masses"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Is the reversed name of this landmark's country a real word?",
            "Answer": "No"
        }}],
    },
    {
        "src": "If the person in this image traveled at the speed of light, how long would it take to reach the nearest star?",
        "rephrase": "Calculate light-speed travel time to the nearest star for the person shown.",
        "pred": "Instant",
        "alt": "4.24 years",
        "loc": "What is the circumference of Earth?",
        "loc_ans": "40,075 km",
        "m_loc_q": "What is the person doing?",
        "m_loc_a": "Standing",
        "textual_edit": {
            "src": "How far is the nearest star from Earth?",
            "rephrase": "What is the distance to Proxima Centauri?",
            "pred": ["Infinitely far"],
            "alt": ["4.24 light years"],
            "loc": "What is the circumference of Earth?",
            "loc_ans": "40,075 km"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "What is the name of the nearest star to Earth besides the Sun?",
            "Answer": "Proxima Centauri"
        }}],
    },
    {
        "src": "What would happen to the building in this image if gravity doubled?",
        "rephrase": "Predict structural effects on this building under 2x gravity.",
        "pred": "Nothing would change",
        "alt": "Structural stress would increase significantly",
        "loc": "What is Avogadro's number?",
        "loc_ans": "6.022 x 10^23",
        "m_loc_q": "What is the building height?",
        "m_loc_a": "About 300 meters",
        "textual_edit": {
            "src": "What happens to buildings if gravity doubles?",
            "rephrase": "Effect of doubled gravity on structures?",
            "pred": ["Buildings are unaffected by gravity changes"],
            "alt": ["All loads double, causing potential collapse"],
            "loc": "What is Avogadro's number?",
            "loc_ans": "6.022 x 10^23"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "By what factor would the foundation stress increase if gravity doubled?",
            "Answer": "2"
        }}],
    },
    {
        "src": "If you combined the population of every city visible in this satellite image, what percentage of Earth's population would that be?",
        "rephrase": "Estimate the fraction of world population in the cities shown in this satellite view.",
        "pred": "50 percent",
        "alt": "Less than 1 percent",
        "loc": "What is entropy?",
        "loc_ans": "A measure of disorder in a system",
        "m_loc_q": "What scale is this satellite image?",
        "m_loc_a": "Regional",
        "textual_edit": {
            "src": "What fraction of Earth's population lives in small cities?",
            "rephrase": "What percentage of world population is in small towns?",
            "pred": ["Most people live in all cities combined"],
            "alt": ["Less than 1 percent in any single small area"],
            "loc": "What is entropy?",
            "loc_ans": "A measure of disorder in a system"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "How many cities are visible in this satellite image?",
            "Answer": "Cannot be determined from a single image"
        }}],
    },
    {
        "src": "If the painting in this image was created one century later, what art movement would it belong to?",
        "rephrase": "Shift the creation date of this painting by 100 years and name the matching art movement.",
        "pred": "Impressionism",
        "alt": "Cubism",
        "loc": "What is the Heisenberg uncertainty principle?",
        "loc_ans": "Cannot simultaneously know position and momentum precisely",
        "m_loc_q": "What colors dominate this painting?",
        "m_loc_a": "Earth tones",
        "textual_edit": {
            "src": "What art movement was dominant in the early 1900s?",
            "rephrase": "Name the art movement of the early 20th century.",
            "pred": ["Impressionism which dominated the 1800s"],
            "alt": ["Cubism which emerged around 1907"],
            "loc": "What is the Heisenberg uncertainty principle?",
            "loc_ans": "Cannot simultaneously know position and momentum precisely"
        },
        "port_new": [{"port_type": "comp", "Q&A": {
            "Question": "Who pioneered the art movement that matches 100 years after this painting?",
            "Answer": "Pablo Picasso"
        }}],
    },
]

# ─────────────────────────────────────────────────────────────────────
# ASSEMBLE FINAL DATASET
# ─────────────────────────────────────────────────────────────────────
all_templates = (
    ambiguity_templates +       # 15
    conflicting_templates +     # 10
    retrieval_trap_templates +   # 10
    noisy_reasoning_templates +  # 10
    extreme_templates            # 5
)

assert len(all_templates) == 50, f"Expected 50, got {len(all_templates)}"

# Assign images from pool
for i, sample in enumerate(all_templates):
    imgs = pick_images(3)
    sample["image"] = imgs[0]
    sample["image_rephrase"] = imgs[1]
    sample["m_loc"] = imgs[2]
    # Add missing keys to match schema
    if "src_q" not in sample:
        sample["src_q"] = None
    if "rephrase_q" not in sample:
        sample["rephrase_q"] = None
    if "m_loc_q_q" not in sample:
        sample["m_loc_q_q"] = None

# Reorder keys to match original schema
ordered_keys = [
    "src", "rephrase", "pred", "alt", "image", "image_rephrase",
    "loc", "loc_ans", "m_loc", "m_loc_q", "m_loc_a",
    "src_q", "rephrase_q", "m_loc_q_q", "port_new", "textual_edit"
]

final_dataset = []
for s in all_templates:
    ordered = {}
    for k in ordered_keys:
        ordered[k] = s.get(k)
    final_dataset.append(ordered)

# Save full dataset
with open("datasets/adversarial_reasoning_dataset.json", "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, indent=2, ensure_ascii=False)

# Save preview
with open("datasets/adversarial_preview.json", "w", encoding="utf-8") as f:
    json.dump(final_dataset[:3], f, indent=2, ensure_ascii=False)

print(f"Generated {len(final_dataset)} adversarial samples")
print(f"  - 15 ambiguity")
print(f"  - 10 conflicting")
print(f"  - 10 retrieval traps")
print(f"  - 10 noisy reasoning")
print(f"  - 5 extreme adversarial")
print(f"Saved: datasets/adversarial_reasoning_dataset.json")
print(f"Saved: datasets/adversarial_preview.json")
