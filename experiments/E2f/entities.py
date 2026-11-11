"""Safe entity question sets for the E2f ignorance direction (no hazardous topics).

Build: 300 questions about very well-known real people / places / books (50 entities per type x 2 templates) and
300 matched questions about made-up ones (same types and templates; names generated from a seeded syllable
generator, checked against the real lists). Held-out: 100 questions about 50 further real entities (17 people,
17 places, 16 books x the same 2 templates), never used to build the direction.
"""
import numpy as np

PEOPLE = ["Albert Einstein", "Isaac Newton", "Marie Curie", "Charles Darwin", "William Shakespeare",
          "Leonardo da Vinci", "Napoleon Bonaparte", "Abraham Lincoln", "George Washington", "Mahatma Gandhi",
          "Nelson Mandela", "Martin Luther King Jr.", "Winston Churchill", "Cleopatra", "Julius Caesar", "Aristotle",
          "Plato", "Socrates", "Galileo Galilei", "Nikola Tesla", "Thomas Edison", "Wolfgang Amadeus Mozart",
          "Ludwig van Beethoven", "Johann Sebastian Bach", "Pablo Picasso", "Vincent van Gogh", "Michelangelo",
          "Queen Victoria", "Elizabeth I", "Alexander the Great", "Genghis Khan", "Christopher Columbus",
          "Neil Armstrong", "Barack Obama", "John F. Kennedy", "Franklin D. Roosevelt", "Karl Marx", "Sigmund Freud",
          "Confucius", "Siddhartha Gautama", "Muhammad Ali", "Michael Jordan", "Pele", "Steve Jobs", "Bill Gates",
          "Elvis Presley", "John Lennon", "Charlie Chaplin", "Walt Disney", "Mark Twain",
          # held-out (17)
          "Charles Dickens", "Jane Austen", "Leo Tolstoy", "Ernest Hemingway", "Stephen Hawking", "Ada Lovelace",
          "Alan Turing", "Florence Nightingale", "Joan of Arc", "Frida Kahlo", "Mother Teresa", "Queen Elizabeth II",
          "Vladimir Lenin", "Rosa Parks", "Amelia Earhart", "Usain Bolt", "Michael Jackson"]
PLACES = ["Paris", "London", "Tokyo", "New York City", "Rome", "Berlin", "Moscow", "Beijing", "Sydney", "Cairo",
          "Rio de Janeiro", "Mumbai", "Istanbul", "Madrid", "Toronto", "Los Angeles", "Chicago", "Venice", "Amsterdam",
          "Athens", "Barcelona", "Dubai", "Singapore", "Hong Kong", "Bangkok", "Mexico City", "Buenos Aires",
          "Cape Town", "Vienna", "Prague", "Dublin", "Lisbon", "Stockholm", "Seoul", "San Francisco", "Kyoto",
          "Florence", "Edinburgh", "Las Vegas", "Shanghai", "Nairobi", "Havana", "Copenhagen", "Oslo", "Budapest",
          "Brussels", "Geneva", "Montreal", "Vancouver", "Boston",
          # held-out (17)
          "Miami", "Lima", "Santiago", "Kathmandu", "Mount Everest", "the Sahara Desert", "the Amazon River",
          "the Grand Canyon", "Niagara Falls", "the Great Barrier Reef", "Lake Victoria", "the Nile River", "Warsaw",
          "Helsinki", "Delhi", "Washington, D.C.", "Machu Picchu"]
BOOKS = ["Pride and Prejudice", "Nineteen Eighty-Four", "To Kill a Mockingbird", "The Great Gatsby", "Moby-Dick",
         "War and Peace", "Don Quixote", "Hamlet", "Romeo and Juliet", "The Odyssey", "The Iliad",
         "Crime and Punishment", "The Catcher in the Rye", "Harry Potter and the Philosopher's Stone",
         "The Lord of the Rings", "The Hobbit", "Jane Eyre", "Wuthering Heights", "Frankenstein", "Dracula",
         "Les Miserables", "The Adventures of Huckleberry Finn", "Alice's Adventures in Wonderland", "Animal Farm",
         "Brave New World", "The Divine Comedy", "One Hundred Years of Solitude", "The Little Prince",
         "Anna Karenina", "Great Expectations", "Oliver Twist", "A Tale of Two Cities", "The Grapes of Wrath",
         "Of Mice and Men", "Lord of the Flies", "Fahrenheit 451", "The Old Man and the Sea", "Gulliver's Travels",
         "Robinson Crusoe", "Treasure Island", "The Count of Monte Cristo", "The Three Musketeers", "Madame Bovary",
         "The Brothers Karamazov", "Ulysses", "Macbeth", "Othello", "King Lear", "A Brief History of Time",
         "The Da Vinci Code",
         # held-out (16)
         "The Hunger Games", "Charlotte's Web", "The Diary of a Young Girl", "The Art of War", "The Prince",
         "The Communist Manifesto", "Little Women", "Sense and Sensibility", "The Picture of Dorian Gray",
         "The Wonderful Wizard of Oz", "Peter Pan", "The Jungle Book", "The Alchemist", "Gone with the Wind",
         "The Chronicles of Narnia", "Strange Case of Dr Jekyll and Mr Hyde"]
TEMPLATES = {"person": ["Who is {x}?", "What is {x} best known for?"],
             "place": ["Where is {x} located?", "What is {x} famous for?"],
             "book": ["Who wrote the book \"{x}\"?", "What is the book \"{x}\" about?"]}
N_BUILD = 50

_SYL = ["ka", "lo", "ven", "dri", "mar", "sel", "tor", "bin", "quo", "zel", "ra", "fen", "mo", "lus", "tre", "vok",
        "nal", "pim", "gar", "shu", "dex", "ol", "yan", "bru", "cel", "thi", "rop", "wen", "ix", "abe"]
_ADJ = ["Silent", "Crimson", "Hollow", "Gilded", "Distant", "Broken", "Velvet", "Hidden", "Ashen", "Wandering"]
_NOUN = ["Lantern", "Orchard", "Harbor", "Mirror", "Compass", "Meadow", "Tower", "Garden", "River", "Clock"]
_PLACE_FMT = ["{n}ville", "Port {n}", "Lake {n}", "{n} Bay", "Mount {n}", "{n}burg", "the {n} Valley", "{n}stad"]


def _name(rng, k):
    return "".join(_SYL[i] for i in rng.choice(len(_SYL), size=k, replace=True)).capitalize()


def made_up(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    real = {x.lower() for x in PEOPLE + PLACES + BOOKS}
    out = {"person": [], "place": [], "book": []}
    seen = set()
    while min(len(v) for v in out.values()) < N_BUILD:
        cands = {"person": f"{_name(rng, 2)} {_name(rng, 3)}",
                 "place": _PLACE_FMT[rng.integers(len(_PLACE_FMT))].format(n=_name(rng, 2)),
                 "book": f"The {_ADJ[rng.integers(10)]} {_NOUN[rng.integers(10)]} of {_name(rng, 3)}"}
        for t, x in cands.items():
            if len(out[t]) < N_BUILD and x.lower() not in real and x not in seen:
                out[t].append(x)
                seen.add(x)
    return out


def _questions(ents: dict, kind: str) -> list[dict]:
    return [{"id": f"{kind}:{t}:{i}:{j}", "type": t, "kind": kind, "entity": x, "question": tpl.format(x=x)}
            for t, xs in ents.items() for i, x in enumerate(xs) for j, tpl in enumerate(TEMPLATES[t])]


def build_sets() -> dict:
    real_b = {"person": PEOPLE[:N_BUILD], "place": PLACES[:N_BUILD], "book": BOOKS[:N_BUILD]}
    held = {"person": PEOPLE[N_BUILD:], "place": PLACES[N_BUILD:], "book": BOOKS[N_BUILD:]}
    s = {"real": _questions(real_b, "real"), "fake": _questions(made_up(), "fake"), "heldout": _questions(held, "heldout")}
    assert len(s["real"]) == 300 and len(s["fake"]) == 300 and len(s["heldout"]) == 100, {k: len(v) for k, v in s.items()}
    assert not ({q["entity"] for q in s["heldout"]} & {q["entity"] for q in s["real"]})
    return s
