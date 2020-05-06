import pandas as pd
from collections import Counter
import pickle
from bs4 import BeautifulSoup

f = open("../colors-in-context/outputs/color_samples.html", "r").read()
soup = BeautifulSoup(f, features="html.parser")

rows = soup.find_all("tr")

rows = [r.find_all("td") for r in rows]

colors = [list(map(lambda x: x.text, r)) for r in rows]
colors = [c for c in colors if len(c) > 0]

colors = {r[0]: r[1:] for r in colors}

pickle.dump(colors, open("./data/colors_probs_better.pkl", "wb"))
