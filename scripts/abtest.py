import argparse
import re
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy.stats import mannwhitneyu, ttest_ind, trim_mean
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--variants', type=str)
parser.add_argument('--variables', type=str, default=None)
parser.add_argument('--labels', type=str, default=None)
parser.add_argument('--colors', type=str, default=None)
parser.add_argument('--test', type=str, default='u')
parser.add_argument('--estimator', type=str, default='mean')
parser.add_argument('--ncols', type=int, default=4)
parser.add_argument('--dpi', type=int, default=None)
parser.add_argument('--savefig', type=str, default='abtest.png')
args = parser.parse_args()
assert args.variants, 'missing flag --variants=<a_path>,<b_path>,...'
test = mannwhitneyu if args.test == 'u' else lambda x, y, alternative: ttest_ind(x, y, equal_var=False, alternative=alternative)
variants = args.variants.split(',')
labels = args.labels.split(',') if args.labels else variants
variables = args.variables.split(',') if args.variables else []
pattern = re.compile(r"^(?P<value>\d+(?:\.\d+)?)\s+(?P<name>.+)$", re.MULTILINE)
colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green'][:len(variants)]

data = {}
for path in variants:
	data[path] = defaultdict(list)
	with open(path, 'r') as f:
		 text = f.read()
	for value, name in pattern.findall(text):
		data[path][name].append(float(value))

if not variables:
	variables = list(data[path].keys())

for a, b in combinations(variants, 2):
	p_values = []
	for v in variables:
		#alternative = 'greater' if np.mean(data[a][v]) > np.mean(data[b][v]) else 'less'
		alternative = 'two-sided'
		#alternative = 'less'
		_, p_value = test(data[a][v], data[b][v], alternative=alternative)
		#print(f"{np.mean(data[a][v]):.4f} {np.mean(data[b][v]):.4f} {p_value:.6f} {v}")
		p_values.append(p_value)
	rejects, p_values, _, _ = multipletests(p_values, method='holm')
	print(a, 'vs', b)
	for v, reject, p_value in zip(variables, rejects, p_values):
		a_mean, a_std = np.mean(data[a][v]), np.std(data[a][v])
		b_mean, b_std = np.mean(data[b][v]), np.std(data[b][v])
		direction = a_mean > b_mean
		diff = 100*(b_mean - a_mean)
		reject = ('>' if direction else '<') if reject else ' '
		print(f'{diff:+.6f}p.p. {a_mean:.6f}±{a_std:.6f} {reject} {b_mean:.6f}±{b_std:.6f} p={p_value:.6f} {v}')
	print()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def percent_formatter(x, pos):
	x *= 100
	if x >= 1:
		return f'{x:.0f}%'
	return f'{x:g}%'

if False:
	plot_data = []
	for i, v in enumerate(variables):
		for path, label in zip(variants, labels):
			v_label = v[2:-1].replace('_', ' ')
			for val in data[path][v]:
				plot_data.append({'Variant': label, 'Variable': v_label, 'Probability': val})
	plot_data = pd.DataFrame(plot_data)
	plt.figure(figsize=(9, 4))
	iqm = lambda x: trim_mean(x, 0.25)
	order = plot_data.groupby('Variable')['Probability'].apply(iqm).sort_values(ascending=False).index
	ax = sns.barplot(data=plot_data, x='Variable', y='Probability', hue='Variant', estimator=iqm, errorbar=('pi', 50), order=order)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=12)

if True:
	num_vars = len(variables)
	cols = args.ncols
	rows = (num_vars + cols - 1) // cols
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=True)
	axes = axes.flatten()

	for i, v in enumerate(variables):
		ax = axes[i]
		# Prepare data for seaborn
		plot_data = []
		for path, label in zip(variants, labels):
			for val in data[path][v]:
				plot_data.append({'Variant': label, 'Value': val})
		df_plot = pd.DataFrame(plot_data)
		
		# Plot Boxplot (IQR)
		#sns.boxplot(data=df_plot, x='Variant', y='Value', ax=ax, hue='Variant', whis=1.5, showfliers=False)
		sns.barplot(data=df_plot, x='Variant', y='Value', ax=ax, hue='Variant', alpha=1, errorbar=None, legend=None, palette=colors, estimator=args.estimator)
			#errorbar='sd', capsize=0.1, legend=False, err_kws={'linewidth': 1, 'alpha': 0.6})
		# Plot Individual Observations
		sns.stripplot(data=df_plot, x='Variant', y='Value', ax=ax, color='black', marker='.', jitter=0.1)
		
		ax.set_title(v.replace('_', ' ').replace('|', '|\n').replace('diamond', 'dia.'), fontsize=10)
		ax.yaxis.set_major_formatter(percent_formatter)
		#ax.set_ylim(0,1)
		ax.set(xticks=[], xticklabels=[])
		ax.set_ylabel('')
		ax.set_xlabel('')

	#fig.subplots_adjust(bottom=0.15)
	plt.tight_layout(rect=[0, 0.10, 1, 1])

	handles = [plt.Line2D([0], [0], color=c, lw=8) for c in colors]
	fig.legend(handles=handles, labels=labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.03), frameon=False)

	# Remove empty subplots
	for j in range(i + 1, len(axes)):
		fig.delaxes(axes[j])

#plt.tight_layout()
plt.savefig(args.savefig, dpi=args.dpi)
print("Plot saved to", args.savefig)
