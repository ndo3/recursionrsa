import pandas as pd
from collections import Counter
import pickle
from bs4 import BeautifulSoup
from functools import reduce

f = open("../colors-in-context/outputs/color_samples.html", "r").read()
soup = BeautifulSoup(f, features="html.parser")

rows = soup.find_all("tr")

rows = [r.find_all("td") for r in rows]

colors = [list(map(lambda x: x.text, r)) for r in rows]
colors = [c for c in colors if len(c) > 0]

hsls = [r[0] for r in colors]

df = pd.DataFrame(index = hsls, columns = ['utterances'])
df['utterances'] = [r[1:] for r in colors]

#pickle.dump(df, open("./data/colors_probs_better.pkl", "wb"))


colors = [(c[0][1:-1].split(", "), c[1:]) for c in colors]
colors = [((int(c[0][0]), int(c[0][1]), int(c[0][2])), c[1]) for c in colors]

colors = [(c[1], c[0]) for c in colors]
colors = [(c_inner, c[1]) for c in colors for c_inner in c[0]]
colors_dict = {}
for c in colors:
    if c[0] in colors_dict:
        colors_dict[c[0]] += [c[1]]
    else:
        colors_dict[c[0]] = [c[1]]

pickle.dump(colors_dict, open("utterances_prob_better.pkl", "wb"))









#pickle.dump(colors, open("./data/colors_probs_better.pkl", "wb"))
