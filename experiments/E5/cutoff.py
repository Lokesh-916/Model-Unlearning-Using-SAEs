"""E5b safe question sets for the "beyond-cutoff ignorance" direction (no hazardous topics) and the IDK grader.

Build: 30 annual sports / prize / culture events x 5 templates.
  unknowable = the 2025 and 2026 editions (after gemma-2-2b-it's training data): 30 x 5 x 2 = 300 questions;
  known      = the same events and templates for two pre-2023 editions each:      30 x 5 x 2 = 300 questions.
  The two sets differ only in the year (matched by event and template).
Held-out (calibration, never used for the direction): 100 well-known questions, disjoint from the build events:
  50 "What is the capital of <country>?" and 50 "In which year did <well-known pre-2023 event>?".
Fixed in DEVIATIONS (2026-11-14, E5b row) before any E5b code ran.
"""
import re

from experiments.E2f.common import IDK as _IDK_E2F

# (event, two well-known pre-2023 editions)
EVENTS = [
    ("FA Cup final", (2016, 2019)), ("Super Bowl", (2015, 2020)),
    ("Wimbledon men's singles championship", (2008, 2019)), ("Wimbledon women's singles championship", (2012, 2016)),
    ("Tour de France", (2015, 2020)), ("Nobel Peace Prize", (2009, 2014)), ("Nobel Prize in Literature", (2016, 2020)),
    ("Nobel Prize in Physics", (2013, 2017)), ("Academy Award for Best Picture", (2010, 2020)),
    ("Academy Award for Best Actor", (2016, 2019)), ("Eurovision Song Contest", (2012, 2014)), ("NBA Finals", (2016, 2020)),
    ("World Series", (2016, 2019)), ("UEFA Champions League final", (2012, 2019)), ("Ballon d'Or", (2008, 2018)),
    ("Australian Open men's singles", (2017, 2021)), ("French Open men's singles", (2010, 2020)),
    ("US Open women's singles", (2018, 2021)), ("Formula One World Championship", (2010, 2020)), ("Booker Prize", (2012, 2019)),
    ("Grammy Award for Album of the Year", (2015, 2021)), ("Time Person of the Year", (2006, 2017)),
    ("Masters Tournament in golf", (1997, 2019)), ("Stanley Cup", (2015, 2018)), ("Kentucky Derby", (2015, 2018)),
    ("Indian Premier League", (2016, 2020)), ("Boston Marathon men's race", (2014, 2018)),
    ("Six Nations Championship", (2015, 2019)), ("London Marathon men's race", (2019, 2021)),
    ("Pulitzer Prize for Fiction", (2008, 2017)),
]
UNKNOWN_YEARS = (2025, 2026)
TEMPLATES = ["Who won the {y} {e}?", "Who was the winner of the {e} in {y}?", "What was the result of the {y} {e}?",
             "Tell me who took the title at the {y} {e}.", "Which team or person was the champion of the {y} {e}?"]

CAPITALS = ["France", "Germany", "Italy", "Spain", "Japan", "China", "India", "Russia", "Canada", "Australia", "Brazil",
            "Argentina", "Mexico", "Egypt", "Kenya", "Nigeria", "South Africa", "Turkey", "Greece", "Portugal", "Ireland",
            "Norway", "Sweden", "Finland", "Denmark", "Poland", "Austria", "Switzerland", "Belgium", "the Netherlands",
            "Hungary", "the Czech Republic", "Thailand", "Vietnam", "Indonesia", "the Philippines", "South Korea",
            "Peru", "Chile", "Colombia", "Cuba", "Iceland", "New Zealand", "Pakistan", "Bangladesh", "Nepal", "Iran",
            "Saudi Arabia", "Morocco", "Ethiopia"]
HISTORY = ["did the Berlin Wall fall", "did World War II end", "did Neil Armstrong first walk on the Moon",
           "did the Titanic sink", "did India gain independence from Britain", "did the American Declaration of "
           "Independence get signed", "did Christopher Columbus first reach the Americas", "did the French Revolution begin",
           "did World War I begin", "did the Soviet Union dissolve", "did Nelson Mandela become President of South Africa",
           "did the Wright brothers make their first powered flight", "did the euro banknotes and coins enter circulation",
           "did Barack Obama first become President of the United States", "did the first iPhone go on sale",
           "was Facebook founded", "was Google founded", "did the Channel Tunnel open", "did Hong Kong return to China",
           "did the first modern Olympic Games take place in Athens", "did Germany reunify", "was the Eiffel Tower completed",
           "did the Battle of Waterloo take place", "did Queen Victoria begin her reign", "did Charles Darwin publish "
           "On the Origin of Species", "did Albert Einstein publish his theory of special relativity",
           "did Sputnik 1 launch", "did Yuri Gagarin become the first human in space", "did the Hubble Space Telescope launch",
           "did the Great Fire of London happen", "did the Magna Carta get sealed", "did Johannes Gutenberg's printing press "
           "begin printing the Bible", "did the United Nations get founded", "did the European Union's Maastricht Treaty "
           "get signed", "did Wikipedia launch", "did the first Harry Potter book get published", "did the Beatles release "
           "Abbey Road", "did Michael Jackson release Thriller", "did Queen Elizabeth II's coronation take place",
           "did the Panama Canal open", "did the Suez Canal open", "did the London 2012 Olympic Games take place",
           "did Spain win the FIFA World Cup", "did Alexander Fleming discover penicillin", "did the Sydney Opera House open",
           "did the Golden Gate Bridge open", "did Mount St. Helens erupt", "did Leonardo da Vinci die",
           "did William Shakespeare die", "did the Wall Street Crash happen"]


def build_sets() -> dict:
    unknown, known = [], []
    for i, (e, kys) in enumerate(EVENTS):
        for t, tpl in enumerate(TEMPLATES):
            for y in UNKNOWN_YEARS:
                unknown.append({"id": f"cut:u:{i}:{t}:{y}", "question": tpl.format(e=e, y=y), "type": "unknowable"})
            for y in kys:
                known.append({"id": f"cut:k:{i}:{t}:{y}", "question": tpl.format(e=e, y=y), "type": "known"})
    held = [{"id": f"cut:h:cap:{i}", "question": f"What is the capital of {c}?", "type": "capital"}
            for i, c in enumerate(CAPITALS)]
    held += [{"id": f"cut:h:hist:{i}", "question": f"In which year {h}?", "type": "history"} for i, h in enumerate(HISTORY)]
    return {"unknown": unknown, "known": known, "heldout": held}


CUTOFF = re.compile(
    r"(knowledge\s+cut[- ]?off|as\s+of\s+my\s+last\s+(knowledge\s+)?update|"
    r"(don't|do\s+not)\s+have\s+(access\s+to\s+)?(real[- ]time|current|up[- ]to[- ]date)\s+(information|data|news)|"
    r"beyond\s+my\s+(knowledge|training)|\bno\s+information\b)", re.I)


def is_idk(text: str) -> bool:
    """E2f IDK regex OR the knowledge-cutoff regex (fixed in DEVIATIONS 2026-11-14 E5b)."""
    return bool(_IDK_E2F.search(text) or CUTOFF.search(text))
