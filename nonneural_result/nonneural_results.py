import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import ast
import pandas as pd

results = open("classical_results.txt", "r").readlines()
results = [r.split("\n")[0] for r in results]
results = [r.split("|") for r in results]
results = [(float(a), ast.literal_eval(d)) for a,d in results]
print(results)
results = [{'alpha': [a]*6, 'recursions': list(range(0, 12, 2)), 'accuracy': [d[str(i)] for i in range(0, 12, 2)]} for a,d in results]
results = {'alpha': [r['alpha'] for r in results],
            'recursions': [r['recursions'] for r in results],
            'accuracy': [r['accuracy'] for r in results]}
results = {'alpha': [aa for a in results['alpha'] for aa in a],
            'recursions': [rr for r in results['recursions'] for rr in r],
            'accuracy': [aa for a in results['accuracy'] for aa in a]}



df_grouped = pd.DataFrame(results)
#df['alpha_recursions'] = list(zip(df['alpha'], df['recursions']))
#df_grouped = df.groupby(['alpha_recursions']).agg('mean')

print(df_grouped)


ax = sns.lineplot(x = "recursions", y = "accuracy", hue = 'alpha', data = df_grouped, legend = "full")
plt.ylim([0.25, 0.7])
plt.title("Classical RSA")
plt.show()
