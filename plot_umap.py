import argparse
import torch
import umap
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from craftax.craftax.constants import BlockType, ItemType, MobType2
from agent import load_agent_from_save

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="models/best.sav")
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--ncols", type=int, default=None)
parser.add_argument("--figsize", type=float, default=4)
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument("--savefig", type=str, default='umap.png')
parser.add_argument("--pca", action='store_true')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--poster", action='store_true')

args = parser.parse_args()
seed = args.seed
models = args.model.split(',')
names = args.name.split(',') if args.name else models
ignore_blocks = {0,6,20,25,26,27,28,29,30,31,32,33,34,35,36,37}
ignore_mobs = {3,4,5,6,7,11,12,13,14,15,19,20,21,22,23,27,28,29,30,31,32,33,34,35,36,37,38,39}

if args.poster:
    plt.rcParams.update({'font.size': 14})

assets = {
    'melee_1': 'zombie.png',
    'melee_2': 'orc_soldier.png',
    'melee_3': 'gnome_warrior.png',
    'ranged_1': 'skeleton.png',
    'ranged_2': 'orc_mage.png',
    'ranged_3': 'gnome_archer.png',
    'passive_1': 'cow.png',
    'passive_2': 'snail.png',
    'passive_3': 'bat.png',
    'enemy_arrow_1': 'arrow-up.png',
    'enemy_arrow_2': 'fireball.png',
    'crafting_table': 'table.png',
    'torch': 'torch_on_path.png',
    'ripe_plant': 'ripe_plant_on_grass.png',
}

colors = {
    'MELEE_1': 'red',
    'MELEE_2': 'red',
    'MELEE_3': 'red',
    'RANGED_1': 'orange',
    'RANGED_2': 'orange',
    'RANGED_3': 'orange',
    'PASSIVE_1': 'green',
    'PASSIVE_2': 'green',
    'PASSIVE_3': 'green',
    'RIPE_PLANT': 'green',
    #'RUBY': 'gray',
    #'SAPPHIRE': 'gray',
    #'DIAMOND': 'gray',
    'IRON': 'gray',
    'COAL': 'gray',
    'STONE': 'gray',
    'CHEST': 'gray',
    'TREE': 'gray',
    #'LADDER_DOWN_BLOCKED': 'gray',
    'LADDER_DOWN': 'gray',
    'CRAFTING_TABLE': 'gray',
    'FURNACE': 'gray',
    #'TORCH': 'gray',
    #'LADDER_UP': 'gray',
    #'TORCH': 'yellow',
    'WATER': 'blue',
    'FOUNTAIN': 'blue',
}

markers = {
    'MELEE_1': 'o',
    'MELEE_2': 'o',
    'MELEE_3': 'o',
    'RANGED_1': 'o',
    'RANGED_2': 'o',
    'RANGED_3': 'o',
    'PASSIVE_1': 'o',
    'PASSIVE_2': 'o',
    'PASSIVE_3': 'o',
    'RIPE_PLANT': 'o',
    #'RUBY': 'd',
    #'SAPPHIRE': 'd',
    #'DIAMOND': 'd',
    'IRON': 'o',
    'COAL': 'o',
    'STONE': 'o',
    'CHEST': 'o',
    #'TREE': 'd',
    #'LADDER_DOWN_BLOCKED': '*',
    #'LADDER_DOWN': '*',
    #'LADDER_UP': '*',
    'WATER': 'o',
    'FOUNTAIN': 'o',
}

tiles = torch.eye(82,83)



"""
reducer = umap.UMAP(random_state=seed)
reducer = PCA(n_components=2)
reduced = reducer.fit_transform(embeddings)
for i, (x, y, label) in enumerate(zip(x, y, labels)):
    if i in ignore_blocks or i-42 in ignore_mobs: continue
    plt.plot(x, y, 'o')
    plt.annotate(label, (x, y), xytext=(0, 0), textcoords='offset points', ha='left')
"""

def plot_images(ax, data, labels):
    ax.set_xticks([])
    ax.set_yticks([])
    for i, (x, y) in enumerate(data):
        if i in ignore_blocks or i-42 in ignore_mobs:
            continue
        label = labels[i].lower()
        png = assets.get(label, label+'.png')
        print(label, png)
        try:
            img = plt.imread('craftax/craftax/assets/'+png)
        except Exception:
            continue
            img = plt.imread('craftax/craftax/assets/debug_tile.png')
        im = OffsetImage(img)
        ab = AnnotationBbox(im, (x, y), frameon=False)
        ax.add_artist(ab)
        #ax.annotate(label, (x, y), xytext=(0, -15), 
        #                textcoords='offset points', ha='center', fontsize=8)
    ax.update_datalim(data)
    ax.autoscale()

labels = []
for type in list(BlockType) + list(ItemType) + list(MobType2):
    labels.append(type.name)

# --- Final Plot ---
ncols = args.ncols if args.ncols else len(models)
from math import ceil
nrows = ceil(len(models) / ncols)
figsize = args.figsize 
fig, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=(figsize*ncols, figsize*nrows))

def scale_vectors_to_bbox(v_matrix):
    """Inputs: v_matrix of shape (N, 2)
    Outputs: Scaled v_matrix of shape (N, 2) in range [-1, 1]
    """
    # Transpose to 2 x N
    V = v_matrix.T  # Shape: (2, N)

    # Compute bounds independently per row (axis=1)
    x_min, x_max = V[0].min(), V[0].max()
    y_min, y_max = V[1].min(), V[1].max()

    # Scale independently to [-1, 1]
    V_scaled = np.empty_like(V)
    V_scaled[0] = 2.0 * (V[0] - x_min) / (x_max - x_min) - 1.0
    V_scaled[1] = 2.0 * (V[1] - y_min) / (y_max - y_min) - 1.0

    # Result transposed back to N x 2
    return V_scaled.T

#plot_images(ax, reduced, labels)
for ax, model, title in zip(axs.flat, models, names):
    agent = load_agent_from_save(model, parallel=1, device='cpu')
    agent.eval()

    labels_to_reduce = []
    tiles_to_reduce = []

    #for tile, label in zip(tiles, labels):
    #    color = colors.get(label, 'black')
    #    if color == 'black':
    #        continue
    #    labels_to_reduce.append(label)
    #    tiles_to_reduce.append(tile)

    #tiles_to_reduce = torch.stack(tiles_to_reduce)

    labels_to_reduce = labels
    tiles_to_reduce = tiles
    embeddings = agent.tile_encoder(tiles_to_reduce).detach().numpy()
    n_neighbors = len(tiles_to_reduce) - 1
    #n_neighbors = 8
    transform = PCA(n_components=2) if args.pca else umap.UMAP(random_state=seed, n_neighbors=n_neighbors)
    reduced = transform.fit_transform(embeddings)

    for i, ((x, y), label) in enumerate(zip(reduced, labels_to_reduce)):
        color = colors.get(label, 'black')
        marker = markers.get(label, 's')
        if color == 'black':
            continue

        lab = label.lower()
        png = assets.get(lab, lab+'.png')
        try:
            img = plt.imread('craftax/craftax/assets/'+png)
        except Exception as err:
            print(err)
            continue
        print(lab, png)
        ax.add_artist(AnnotationBbox(OffsetImage(img), (x, y), frameon=False))
        #ax.grid(True, color='lightgray', zorder=0)
        ax.plot(x, y, marker='s', markersize=20, color=color)
        ax.set_title(title)
        #ax.set_axis_off()
        #for spine in ax.spines.values():
        #    spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig(args.savefig, dpi=args.dpi)
plt.show()
