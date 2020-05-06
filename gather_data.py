import pandas as pd
from collections import Counter
import pickle

df = pd.read_csv("./data/filteredCorpus.csv")

df_click = df[df['clickStatus'] == 'target']
df_click = df_click[['clickColH', 'clickColS', 'clickColL', 'contents']]
df_click.columns = ['colH', 'colS', 'colL', 'contents']

df_alt1 = df[df['alt1Status'] == 'target']
df_alt1 = df_alt1[['alt1ColH', 'alt1ColS', 'alt1ColL', 'contents']]
df_alt1.columns = ['colH', 'colS', 'colL', 'contents']

df_alt2 = df[df['alt2Status'] == 'target']
df_alt2 = df_alt2[['alt2ColH', 'alt2ColS', 'alt2ColL', 'contents']]
df_alt2.columns = ['colH', 'colS', 'colL', 'contents']


df_full = df_click.append(df_alt1).append(df_alt2)

df_full['color'] = list(zip(df_full.colH, df_full.colS, df_full.colL))

df_grouped = df_full.groupby('color')['contents'].apply(list)

colors = list(set(df_full['contents']))
print("made colors")


def make_counters(x):
    colors_counter = Counter()
    colors_counter.update({c:0 for c in colors})
    colors_counter.update(x)
    #denom = len(x)
    #for key in colors_counter:
    #    colors_counter[key] /= denom
    return colors_counter



#df_final = df_grouped.apply(lambda x: make_counters(x))
df_final = df_grouped.reset_index()
print(df_final.head())
pickle.dump(df_final, open("./data/colors_probs.pkl", "wb"))
print("dumped")
