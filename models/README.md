# Agent files

Filename format: `<name>_s<seed>.sav`.

- Baseline (b)
- Compression (e32)
- Autoencoder (a32)
- Shuffled (rand32)
- Grounding (kb32)

## Selected agents

- e32_s16.sav (good)
	- max score = 25.27%
	- P(make iron pickaxe) = 11%
	- P(make iron sword) = 27%
	- P(eat bat|enter gnomish mines) = 70%
	- P(defeat gnome warrior|enter gnomish mines) = 67%
	- P(defeat gnome archer|enter gnomish mines) = 28%
- e32_s18.sav (bad)
	- max score = 23.50%
	- P(make iron pickaxe/sword) = 3% (bad)
	- P(eat bat|enter gnomish mines) = 50%
	- P(defeat gnome warrior|enter gnomish mines) = 8% (bad)
	- P(defeat gnome archer|enter gnomish mines) = 0% (bad)
- a32_s7.sav (good)
	- max score = 26.15%
	- P(make iron pickaxe) = 13%
	- P(make iron sword) = 17%
	- P(eat bat|enter gnomish mines) = 46%
	- P(defeat gnome warrior|enter gnomish mines) = 69%
	- P(defeat gnome archer|enter gnomish mines) = 36%
- a32_s3.sav (bad)
	- max score = 22.61%
	- P(make iron pickaxe) = 3% (bad)
	- P(make iron sword) = 3% (bad)
	- P(eat bat|enter gnomish mines) = 6% (bad)
	- P(defeat gnome warrior|enter gnomish mines) = 8% (bad)
	- P(defeat gnome archer|enter gnomish mines) = 2% (bad)
- kb32_rand_s6.sav (good)
	- max score = 26.59%
	- P(make iron pickaxe) = 11%
	- P(make iron sword) = 20%
	- P(eat bat|enter gnomish mines) = 41%
	- P(defeat gnome warrior|enter gnomish mines) = 42%
	- P(defeat gnome archer|enter gnomish mines) = 19%
- kb32_rand_s14.sav (bad)
	- max score = 23.05%
	- P(make iron pickaxe/sword) = 3% (bad)
	- P(eat bat|enter gnomish mines) = 6% (bad)
	- P(defeat gnome warrior|enter gnomish mines) = 1% (bad)
	- P(defeat gnome archer|enter gnomish mines) = 1% (bad)
- kb32_s8.sav (good)
	- max score = 27.48%
	- P(make iron pickaxe) = 11%
	- P(make iron sword) = 25%
	- P(eat bat|enter gnomish mines) = 66%
	- P(defeat gnome warrior|enter gnomish mines) = 69%
	- P(defeat gnome archer|enter gnomish mines) = 34%
- kb32_s9.sav (bad)
	- max score = 25.27%
	- P(make iron pickaxe/sword) = 3% (bad)
	- P(eat bat|enter gnomish mines) = 66%
	- P(defeat gnome warrior|enter gnomish mines) = 62%
	- P(defeat gnome archer|enter gnomish mines) = 39%

