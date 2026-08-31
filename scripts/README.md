Train and evaluate multiple agents on slurm with:

```
sbatch scripts/train.sh
sbatch -d afterok:$TRAIN_JOB_ID scripts/roll.sh
```

But first:

- set `PYTHON` to the path of your Python binary in `train.sh` and `roll.sh`
- set "lambda k" with `--ground=0.5`
- run a single seed with `SBATCH --array=42`
- parallelize with `SBATCH --array=1-20%4`
- set the model name and agent code using the `NAME` and `AGENT` variables

Generate Figure 3 with:

```
python scripts/abtest.py --dpi=300 --ncols=4 --labels="Baseline,Compression,Autoencoder,Grounding"  --variants="logs/b.roll.out,logs/e32.roll.out,logs/a32.roll.out,logs/kb32.roll.out" --variables="P(enter_dungeon),P(eat_snail|enter_dungeon),P(defeat_orc_soldier|enter_dungeon),P(defeat_orc_mage|enter_dungeon),P(enter_gnomish_mines),P(eat_bat|enter_gnomish_mines),P(defeat_gnome_warrior|enter_gnomish_mines),P(defeat_gnome_archer|enter_gnomish_mines)"
```

Generate Figure 4 with:

```
python scripts/abtest.py --dpi=300 --ncols=4 --variants="logs/b.roll.out,logs/e32.roll.out,logs/a32.roll.out,logs/kb32.roll.out" --labels="Baseline,Compression,Autoencoder,Grounding" --variables="P(collect_iron),P(make_iron_sword),P(make_iron_pickaxe),P(make_iron_armour)" 
```

Generate Figure 5 with:

```
python scripts/plot_methods.py --dpi=300 --rolling_window=40 --a="$(echo logs/b_s*.out)" --a_name=Baseline --b="$(echo logs/e32_s*.out)" --b_name=Compression --c="$(echo logs/a32_s*.out)" --c_name=Autoencoder --d="$(echo logs/kb32_s*.out)" --d_name=Grounding
```

Generate Figure 6 and Table 2 with:

```
python scripts/abtest.py --dpi=300 --ncols=6 --labels="Baseline,Compression,Autoencoder,Grounding"  --variants="logs/b.roll.out,logs/e32.roll.out,logs/a32.roll.out,logs/kb32.roll.out" --variables="P(collect_wood),P(make_wood_sword),P(make_wood_pickaxe),P(place_stone),P(place_table),P(place_furnace),P(collect_stone),P(make_stone_sword),P(make_stone_pickaxe),P(collect_coal),P(make_torch),P(place_torch),P(collect_iron),P(make_iron_sword),P(make_iron_pickaxe),P(make_iron_armour),P(open_chest|enter_dungeon),P(find_bow),P(collect_drink),P(eat_plant),P(collect_sapling),P(place_plant),P(make_arrow),P(fire_bow),P(wake_up),P(eat_cow),P(defeat_zombie),P(defeat_skeleton),P(collect_ruby),P(collect_sapphire),P(enter_dungeon),P(eat_snail|enter_dungeon),P(defeat_orc_soldier|enter_dungeon),P(defeat_orc_mage|enter_dungeon),P(collect_diamond),P(make_diamond_sword),P(enter_gnomish_mines),P(eat_bat|enter_gnomish_mines),P(defeat_gnome_warrior|enter_gnomish_mines),P(defeat_gnome_archer|enter_gnomish_mines),P(make_diamond_armour),P(make_diamond_pickaxe)" 
```

The order for Table 2 is:

```
--variables="P(collect_wood),P(make_wood_sword),P(make_wood_pickaxe),P(place_stone),P(place_table),P(place_furnace),P(collect_stone),P(make_stone_sword),P(make_stone_pickaxe),P(collect_coal),P(make_torch),P(place_torch),P(collect_iron),P(make_iron_sword),P(make_iron_pickaxe),P(make_iron_armour),P(find_bow),P(make_arrow),P(fire_bow),P(collect_drink),P(collect_sapling),P(place_plant),P(eat_plant),P(wake_up),P(eat_cow),P(defeat_zombie),P(defeat_skeleton),P(enter_dungeon),P(open_chest|enter_dungeon),P(eat_snail|enter_dungeon),P(defeat_orc_soldier|enter_dungeon),P(defeat_orc_mage|enter_dungeon),P(enter_gnomish_mines),P(eat_bat|enter_gnomish_mines),P(defeat_gnome_warrior|enter_gnomish_mines),P(defeat_gnome_archer|enter_gnomish_mines),P(collect_ruby),P(collect_sapphire),P(collect_diamond),P(make_diamond_sword),P(make_diamond_pickaxe),P(make_diamond_armour)"
```

UMAP for the poster:

```
python plot_umap.py --dpi=400 --poster --model models/e32_s16.sav,models/a32_s7.sav,models/kb32_rand_s6.sav,models/kb32_s8.sav,models/e32_s18.sav,models/a32_s3.sav,models/kb32_rand_s14.sav,models/kb32_s9.sav --figsize=3 --ncols=4 --name "Compression (good),Autoencoder (good),Shuffled (good),Grounding (good),Compression (bad),Autoencoder (bad),Shuffled (bad),Grounding (bad)"
```
