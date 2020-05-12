import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import ast
import pandas as pd

results = open("result_newer.txt", "r").readlines()
results = [r.split("\n")[0] for r in results]
results = [r.split("|") for r in results]
results = [(float(a), ast.literal_eval(d)) for a,d in results]
results = [{'alpha': [a]*6, 'recursions': list(range(0, 12, 2)), 'accuracy': [d[i] for i in range(0, 12, 2)]} for a,d in results]
results = {'alpha': [r['alpha'] for r in results],
            'recursions': [r['recursions'] for r in results],
            'accuracy': [r['accuracy'] for r in results]}
results = {'alpha': [aa for a in results['alpha'] for aa in a],
            'recursions': [rr for r in results['recursions'] for rr in r],
            'accuracy': [aa for a in results['accuracy'] for aa in a]}



df = pd.DataFrame(results)
df['alpha_recursions'] = list(zip(df['alpha'], df['recursions']))
df_grouped = df.groupby(['alpha_recursions']).agg('mean')

print(df_grouped)

palette = sns.color_palette("muted", 5)
ax = sns.lineplot(x = "recursions", y = "accuracy", hue = 'alpha', data = df_grouped, legend = "full", palette = palette)
plt.show()
