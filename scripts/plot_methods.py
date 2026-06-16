import argparse
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np

batch_pattern = re.compile(r"(?P<step>\d+.\d+)M BATCH (?P<batch>\d+)/\d+ .*? REWARD (?P<reward>[\d.]+)")
achievement_pattern = re.compile(r"^\s*(?P<pct>[\d.]+)\s+(?P<name>[a-z_]+)$", re.MULTILINE)

def read_log(path, rolling_window):
	result = []
	
	print('read log', path)
	with open(path, 'r') as f:
		blocks = f.read().split('STEP ')

	last_batch = 0
	for block in blocks:
		match = batch_pattern.search(block)
		if not match: continue

		batch = int(match.group('batch'))
		step = float(match.group('step'))
		reward = float(match.group('reward'))*100/226

		if batch != last_batch + 1:
			print('missing', last_batch + 1)
		last_batch = batch

		result.append({'batch': batch, 'key': 'step', 'val': step})
		result.append({'batch': batch, 'key': 'reward', 'val': reward})
		for percent, name in achievement_pattern.findall(block):
			result.append({'batch': batch, 'key': name, 'val': float(percent)})

	df = pd.DataFrame(result)
	df = df.pivot_table(index='batch', columns='key', values='val')
	df = df.fillna(0).rolling(window=rolling_window).mean()
	return df

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--savefig', type=str, default='training.png')
	parser.add_argument('--rolling_window', type=int, default=10)
	parser.add_argument('--dpi', type=int, default=300)
	variants = ['a', 'b', 'c', 'd']
	for x in variants:
		parser.add_argument(f'--{x}', type=str)
		parser.add_argument(f'--{x}_name', type=str, default=x)
	args = parser.parse_args()
	pd.set_option('display.max_rows', None)

	variants = [x for x in [args.a, args.b, args.c, args.d] if x]
	labels = [args.a_name, args.b_name, args.c_name, args.d_name][:len(variants)]
	colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green'][:len(variants)]

	keys = [
		'collect_wood',
		'make_wood_sword',
		'make_wood_pickaxe',
		'place_stone',
		'place_table',
		'place_furnace',

		'collect_stone',
		'make_stone_sword',
		'make_stone_pickaxe',
		'collect_coal',
		'make_torch',
		'place_torch',

		'collect_iron',
		'make_iron_sword',
		'make_iron_pickaxe',
		'make_iron_armour',
		'open_chest',
		'find_bow',

		'collect_drink',
		'eat_plant',
		'collect_sapling',
		'place_plant',
		'make_arrow',
		'fire_bow',

		'wake_up',
		'eat_cow',
		'defeat_zombie',
		'defeat_skeleton',
		'collect_ruby',
		'collect_sapphire',

		'enter_dungeon',
		'eat_snail',
		'defeat_orc_solider',
		'defeat_orc_mage',
		'collect_diamond',
		'make_diamond_sword',

		'enter_gnomish_mines',
		'eat_bat',
		'defeat_gnome_warrior',
		'defeat_gnome_archer',
		'make_diamond_armour',
		'make_diamond_pickaxe',
	]

	methods = []
	for paths in variants:
		print(paths)
		methods.append([read_log(path, args.rolling_window) for path in paths.split()])

	n_cols = 6
	n_rows = len(keys)//n_cols
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2), sharex=True, squeeze=True)
	for ax, key in reversed(list(zip(axes.flatten(), keys))):
		handles = []
		for i, tables in enumerate(methods):
			repeats = []
			for table in tables:
				if key not in table.columns: continue
				repeats.append(np.array(table[key]))
			if not repeats: continue

			try:
				array = np.stack(repeats)
			except Exception as error:
				for array in repeats:
					print(array.shape)
				raise error

			mean = np.mean(array, axis=0)
			std = np.std(array, axis=0)
			index = 500 * table.index / table.index.max()

			ax.fill_between(index, np.maximum(0, mean-std), mean+std, color=colors[i], linewidth=0, alpha=0.2)
			line, = ax.plot(index, mean, label=labels[i], color=colors[i], linewidth=2, alpha=0.8)
			handles.append(line)

		ax.set_title('P('+key.replace('_', ' ').replace('diamond','dia.')+')', fontsize=10)
		#ax.grid(True, alpha=0.3)

	yformat = lambda x, pos: f"{int(x)}%" if x.is_integer() else f"{x:.1f}%"
	xformat = lambda x, pos: f"{int(x)}M"
	for ax in axes.flatten():
		ax.set_xticks([0, 500])
		ax.xaxis.set_major_formatter(ticker.FuncFormatter(xformat))
		ax.yaxis.set_major_formatter(ticker.FuncFormatter(yformat))

	fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize='medium')
	plt.tight_layout(rect=[0, 0.025, 1, 1])
	plt.subplots_adjust(hspace=0.7, wspace=0.7)
	plt.savefig(args.savefig, dpi=args.dpi)
	plt.show()

	#dfs = [x for x in [a, b, c, d] if x is not None]
	#result = pd.concat(dfs, axis=1)


