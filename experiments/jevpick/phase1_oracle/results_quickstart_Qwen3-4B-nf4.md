# Oracle Study v2: oracle_v2_Qwen3-4B-nf4

推定 speedup = replay (候補なし位置は通常 decode) × 実測 forward コスト比 (HF eager, ctx=1024)。

## G0 表 (L=4, oracle recall@16) と mean max match

| Group | ngram | grammar | corpus | schema | finite | dflash | union |
|---|---:|---:|---:|---:|---:|---:|---:|
| toolcall/test/known | 21.7% / 1.81 | 10.8% / 0.89 | 68.8% / 2.93 | 57.5% / 2.59 | 82.1% / 3.35 | 75.3% / 3.19 | 93.2% / 3.61 |
| toolcall/test/unknown | 20.0% / 1.67 | 9.5% / 0.81 | 28.7% / 1.69 | 54.7% / 2.52 | 61.7% / 2.79 | 69.8% / 3.07 | 83.0% / 3.43 |
| toolcall/train | 20.9% / 1.74 | 10.0% / 0.84 | 62.2% / 2.71 | 56.0% / 2.51 | 76.2% / 3.16 | 72.5% / 3.11 | 89.2% / 3.54 |

## 推定 speedup (oracle / prior top-1), source × L

| Group | Source | L=2 | L=4 | L=8 | L=16 |
|---|---|---:|---:|---:|---:|
| toolcall/test/known | ngram | 1.63x / 1.33x | 1.95x / 1.51x | 1.94x / 1.54x | 1.90x / 1.49x |
| toolcall/test/known | grammar | 1.38x / 1.25x | 1.45x / 1.31x | 1.54x / 1.25x | 1.52x / 1.23x |
| toolcall/test/known | corpus | 1.93x / 1.76x | 2.64x / 2.18x | 3.26x / 2.72x | 3.81x / 2.89x |
| toolcall/test/known | schema | 1.86x / 1.66x | 2.44x / 1.97x | 2.81x / 2.27x | 3.11x / 2.42x |
| toolcall/test/known | finite | 2.07x / 1.83x | 3.00x / 2.30x | 4.04x / 3.03x | 4.90x / 3.22x |
| toolcall/test/known | dflash | 1.99x / 1.96x | 2.66x / 2.64x | 3.27x / 3.21x | 3.34x / 3.30x |
| toolcall/test/known | union | 2.14x / 1.96x | 3.23x / 2.64x | 4.52x / 3.21x | 5.59x / 3.30x |
| toolcall/test/unknown | ngram | 1.58x / 1.30x | 1.80x / 1.43x | 1.85x / 1.47x | 1.80x / 1.43x |
| toolcall/test/unknown | grammar | 1.33x / 1.22x | 1.40x / 1.27x | 1.47x / 1.23x | 1.46x / 1.21x |
| toolcall/test/unknown | corpus | 1.53x / 1.34x | 1.74x / 1.50x | 1.90x / 1.56x | 1.88x / 1.55x |
| toolcall/test/unknown | schema | 1.82x / 1.63x | 2.27x / 1.94x | 2.66x / 2.23x | 2.88x / 2.36x |
| toolcall/test/unknown | finite | 1.87x / 1.60x | 2.40x / 1.90x | 2.92x / 2.19x | 3.12x / 2.27x |
| toolcall/test/unknown | dflash | 1.95x / 1.90x | 2.61x / 2.53x | 3.17x / 3.02x | 3.35x / 3.16x |
| toolcall/test/unknown | union | 2.09x / 1.90x | 2.97x / 2.53x | 3.87x / 3.02x | 4.41x / 3.16x |
| toolcall/train | ngram | 1.59x / 1.31x | 1.83x / 1.47x | 1.84x / 1.50x | 1.80x / 1.47x |
| toolcall/train | grammar | 1.35x / 1.25x | 1.42x / 1.30x | 1.50x / 1.23x | 1.48x / 1.21x |
| toolcall/train | corpus | 1.82x / 1.66x | 2.33x / 1.98x | 2.86x / 2.40x | 3.16x / 2.56x |
| toolcall/train | schema | 1.80x / 1.65x | 2.31x / 1.96x | 2.61x / 2.23x | 2.84x / 2.40x |
| toolcall/train | finite | 1.98x / 1.80x | 2.74x / 2.21x | 3.56x / 2.65x | 4.01x / 2.86x |
| toolcall/train | dflash | 1.97x / 1.92x | 2.64x / 2.57x | 3.23x / 3.12x | 3.33x / 3.22x |
| toolcall/train | union | 2.13x / 1.92x | 3.13x / 2.57x | 4.44x / 3.12x | 5.30x / 3.22x |

## 詳細

| Group | Source | L | n_pos | recall@1 | recall@4 | recall@16 | mean max@16 | mean top-1 | zero-hit@16 | 候補あり率 | 平均候補数 | speedup oracle | speedup top-1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toolcall/test/known | ngram | 2 | 618 | 28.9% | 45.0% | 53.8% | 1.30 | 0.80 | 21.8% | 94.2% | 5.7 | 1.63x | 1.33x |
| toolcall/test/known | ngram | 4 | 618 | 10.4% | 20.4% | 21.7% | 1.81 | 1.07 | 22.0% | 94.2% | 6.5 | 1.95x | 1.51x |
| toolcall/test/known | ngram | 8 | 618 | 0.6% | 1.3% | 1.3% | 2.02 | 1.13 | 22.3% | 94.2% | 7.8 | 1.94x | 1.54x |
| toolcall/test/known | ngram | 16 | 618 | 0.0% | 0.0% | 0.0% | 2.02 | 1.13 | 22.2% | 94.2% | 8.4 | 1.90x | 1.49x |
| toolcall/test/known | grammar | 2 | 618 | 16.7% | 23.4% | 23.4% | 0.63 | 0.54 | 59.2% | 49.8% | 1.5 | 1.38x | 1.25x |
| toolcall/test/known | grammar | 4 | 618 | 3.6% | 10.8% | 10.8% | 0.89 | 0.67 | 59.2% | 49.8% | 1.7 | 1.45x | 1.31x |
| toolcall/test/known | grammar | 8 | 618 | 0.0% | 0.0% | 0.0% | 0.93 | 0.64 | 59.2% | 49.8% | 1.8 | 1.54x | 1.25x |
| toolcall/test/known | grammar | 16 | 618 | 0.0% | 0.0% | 0.0% | 0.93 | 0.64 | 59.2% | 49.8% | 1.8 | 1.52x | 1.23x |
| toolcall/test/known | corpus | 2 | 618 | 68.1% | 75.9% | 79.4% | 1.63 | 1.44 | 13.8% | 92.1% | 21.3 | 1.93x | 1.76x |
| toolcall/test/known | corpus | 4 | 618 | 51.4% | 62.0% | 68.8% | 2.93 | 2.41 | 13.9% | 92.1% | 30.8 | 2.64x | 2.18x |
| toolcall/test/known | corpus | 8 | 618 | 31.4% | 45.2% | 57.7% | 4.92 | 3.62 | 13.8% | 92.1% | 41.2 | 3.26x | 2.72x |
| toolcall/test/known | corpus | 16 | 618 | 17.0% | 27.0% | 39.0% | 7.14 | 4.70 | 13.9% | 92.1% | 56.1 | 3.81x | 2.89x |
| toolcall/test/known | schema | 2 | 618 | 51.7% | 68.2% | 70.2% | 1.48 | 1.17 | 20.4% | 86.6% | 3.1 | 1.86x | 1.66x |
| toolcall/test/known | schema | 4 | 618 | 39.1% | 55.4% | 57.5% | 2.59 | 1.94 | 20.9% | 86.6% | 4.9 | 2.44x | 1.97x |
| toolcall/test/known | schema | 8 | 618 | 23.8% | 36.6% | 37.0% | 4.01 | 2.91 | 20.7% | 86.6% | 6.8 | 2.81x | 2.27x |
| toolcall/test/known | schema | 16 | 618 | 1.6% | 4.1% | 4.4% | 5.09 | 3.44 | 20.7% | 86.6% | 7.9 | 3.11x | 2.42x |
| toolcall/test/known | finite | 2 | 618 | 70.9% | 84.9% | 90.3% | 1.81 | 1.50 | 6.0% | 99.7% | 31.6 | 2.07x | 1.83x |
| toolcall/test/known | finite | 4 | 618 | 53.6% | 69.5% | 82.1% | 3.35 | 2.55 | 6.5% | 99.7% | 43.9 | 3.00x | 2.30x |
| toolcall/test/known | finite | 8 | 618 | 30.5% | 49.0% | 70.3% | 5.76 | 3.82 | 6.3% | 99.7% | 57.6 | 4.04x | 3.03x |
| toolcall/test/known | finite | 16 | 618 | 17.0% | 27.0% | 39.9% | 7.77 | 4.86 | 6.3% | 99.7% | 74.2 | 4.90x | 3.22x |
| toolcall/test/known | dflash | 2 | 618 | 84.3% | 85.1% | 85.1% | 1.76 | 1.73 | 6.5% | 100.0% | 4.0 | 1.99x | 1.96x |
| toolcall/test/known | dflash | 4 | 618 | 74.7% | 75.3% | 75.3% | 3.19 | 3.15 | 6.5% | 100.0% | 4.0 | 2.66x | 2.64x |
| toolcall/test/known | dflash | 8 | 618 | 44.1% | 44.4% | 44.4% | 5.05 | 5.00 | 6.5% | 100.0% | 4.0 | 3.27x | 3.21x |
| toolcall/test/known | dflash | 16 | 618 | 0.0% | 0.0% | 0.0% | 6.03 | 5.98 | 6.5% | 100.0% | 4.0 | 3.34x | 3.30x |
| toolcall/test/known | union | 2 | 618 | 84.3% | 85.1% | 95.5% | 1.89 | 1.73 | 3.2% | 100.0% | 12.1 | 2.14x | 1.96x |
| toolcall/test/known | union | 4 | 618 | 74.7% | 75.3% | 93.2% | 3.61 | 3.15 | 3.2% | 100.0% | 13.3 | 3.23x | 2.64x |
| toolcall/test/known | union | 8 | 618 | 44.1% | 44.4% | 79.1% | 6.41 | 5.00 | 3.2% | 100.0% | 13.9 | 4.52x | 3.21x |
| toolcall/test/known | union | 16 | 618 | 0.0% | 0.0% | 34.9% | 8.99 | 5.98 | 3.2% | 100.0% | 14.4 | 5.59x | 3.30x |
| toolcall/test/unknown | ngram | 2 | 660 | 25.9% | 41.2% | 50.6% | 1.23 | 0.74 | 26.1% | 91.5% | 5.8 | 1.58x | 1.30x |
| toolcall/test/unknown | ngram | 4 | 660 | 11.2% | 19.7% | 20.0% | 1.67 | 1.01 | 26.8% | 91.5% | 6.8 | 1.80x | 1.43x |
| toolcall/test/unknown | ngram | 8 | 660 | 0.0% | 1.5% | 1.5% | 1.91 | 1.07 | 27.0% | 91.5% | 8.3 | 1.85x | 1.47x |
| toolcall/test/unknown | ngram | 16 | 660 | 0.0% | 0.0% | 0.0% | 1.90 | 1.08 | 27.4% | 91.5% | 9.3 | 1.80x | 1.43x |
| toolcall/test/unknown | grammar | 2 | 660 | 15.0% | 21.1% | 21.1% | 0.58 | 0.49 | 62.3% | 47.3% | 1.4 | 1.33x | 1.22x |
| toolcall/test/unknown | grammar | 4 | 660 | 3.2% | 9.5% | 9.5% | 0.81 | 0.60 | 62.3% | 47.3% | 1.5 | 1.40x | 1.27x |
| toolcall/test/unknown | grammar | 8 | 660 | 0.0% | 0.0% | 0.0% | 0.84 | 0.59 | 62.3% | 47.3% | 1.7 | 1.47x | 1.23x |
| toolcall/test/unknown | grammar | 16 | 660 | 0.0% | 0.0% | 0.0% | 0.84 | 0.59 | 62.3% | 47.3% | 1.7 | 1.46x | 1.21x |
| toolcall/test/unknown | corpus | 2 | 660 | 38.1% | 44.2% | 47.5% | 1.09 | 0.89 | 37.4% | 79.7% | 20.0 | 1.53x | 1.34x |
| toolcall/test/unknown | corpus | 4 | 660 | 20.5% | 24.0% | 28.7% | 1.69 | 1.28 | 37.7% | 79.7% | 28.9 | 1.74x | 1.50x |
| toolcall/test/unknown | corpus | 8 | 660 | 5.8% | 8.7% | 13.8% | 2.22 | 1.49 | 37.9% | 79.7% | 38.4 | 1.90x | 1.56x |
| toolcall/test/unknown | corpus | 16 | 660 | 3.9% | 4.2% | 7.5% | 2.73 | 1.76 | 38.6% | 79.7% | 52.1 | 1.88x | 1.55x |
| toolcall/test/unknown | schema | 2 | 660 | 51.7% | 67.5% | 68.8% | 1.44 | 1.16 | 22.3% | 83.2% | 3.3 | 1.82x | 1.63x |
| toolcall/test/unknown | schema | 4 | 660 | 36.7% | 51.7% | 54.7% | 2.52 | 1.88 | 22.6% | 83.2% | 5.5 | 2.27x | 1.94x |
| toolcall/test/unknown | schema | 8 | 660 | 21.0% | 34.0% | 34.0% | 3.85 | 2.80 | 22.7% | 83.2% | 8.1 | 2.66x | 2.23x |
| toolcall/test/unknown | schema | 16 | 660 | 4.2% | 6.4% | 6.7% | 4.92 | 3.29 | 22.7% | 83.2% | 9.3 | 2.88x | 2.36x |
| toolcall/test/unknown | finite | 2 | 660 | 54.8% | 65.0% | 77.3% | 1.60 | 1.24 | 14.7% | 97.6% | 30.4 | 1.87x | 1.60x |
| toolcall/test/unknown | finite | 4 | 660 | 35.8% | 41.5% | 61.7% | 2.79 | 1.96 | 15.0% | 97.6% | 42.8 | 2.40x | 1.90x |
| toolcall/test/unknown | finite | 8 | 660 | 13.3% | 17.3% | 41.5% | 4.35 | 2.62 | 15.5% | 97.6% | 56.5 | 2.92x | 2.19x |
| toolcall/test/unknown | finite | 16 | 660 | 3.9% | 4.2% | 12.2% | 4.49 | 2.95 | 18.3% | 97.6% | 72.5 | 3.12x | 2.27x |
| toolcall/test/unknown | dflash | 2 | 660 | 80.0% | 81.9% | 81.9% | 1.73 | 1.67 | 6.5% | 100.0% | 4.0 | 1.95x | 1.90x |
| toolcall/test/unknown | dflash | 4 | 660 | 69.0% | 69.8% | 69.8% | 3.07 | 3.00 | 6.5% | 100.0% | 4.0 | 2.61x | 2.53x |
| toolcall/test/unknown | dflash | 8 | 660 | 42.5% | 42.7% | 42.7% | 4.84 | 4.75 | 6.5% | 100.0% | 4.0 | 3.17x | 3.02x |
| toolcall/test/unknown | dflash | 16 | 660 | 0.0% | 0.0% | 0.0% | 5.89 | 5.79 | 6.5% | 100.0% | 4.0 | 3.35x | 3.16x |
| toolcall/test/unknown | union | 2 | 660 | 80.0% | 81.9% | 92.0% | 1.86 | 1.67 | 3.6% | 100.0% | 11.7 | 2.09x | 1.90x |
| toolcall/test/unknown | union | 4 | 660 | 69.0% | 69.8% | 83.0% | 3.43 | 3.00 | 3.6% | 100.0% | 12.8 | 2.97x | 2.53x |
| toolcall/test/unknown | union | 8 | 660 | 42.5% | 42.7% | 57.3% | 5.66 | 4.75 | 3.6% | 100.0% | 13.3 | 3.87x | 3.02x |
| toolcall/test/unknown | union | 16 | 660 | 0.0% | 0.0% | 6.1% | 7.16 | 5.79 | 3.6% | 100.0% | 13.8 | 4.41x | 3.16x |
| toolcall/train | ngram | 2 | 2500 | 28.2% | 44.3% | 52.7% | 1.25 | 0.77 | 26.0% | 91.0% | 5.3 | 1.59x | 1.31x |
| toolcall/train | ngram | 4 | 2500 | 11.7% | 20.0% | 20.9% | 1.74 | 1.06 | 26.4% | 91.0% | 6.0 | 1.83x | 1.47x |
| toolcall/train | ngram | 8 | 2500 | 1.4% | 2.7% | 2.8% | 1.98 | 1.15 | 26.5% | 91.0% | 7.0 | 1.84x | 1.50x |
| toolcall/train | ngram | 16 | 2500 | 0.0% | 0.0% | 0.0% | 2.02 | 1.17 | 26.4% | 91.0% | 7.5 | 1.80x | 1.47x |
| toolcall/train | grammar | 2 | 2500 | 15.7% | 21.9% | 21.9% | 0.60 | 0.51 | 61.3% | 48.3% | 1.4 | 1.35x | 1.25x |
| toolcall/train | grammar | 4 | 2500 | 3.4% | 10.0% | 10.0% | 0.84 | 0.64 | 61.3% | 48.3% | 1.6 | 1.42x | 1.30x |
| toolcall/train | grammar | 8 | 2500 | 0.0% | 0.0% | 0.0% | 0.87 | 0.60 | 61.3% | 48.3% | 1.7 | 1.50x | 1.23x |
| toolcall/train | grammar | 16 | 2500 | 0.0% | 0.0% | 0.0% | 0.87 | 0.60 | 61.3% | 48.3% | 1.7 | 1.48x | 1.21x |
| toolcall/train | corpus | 2 | 2500 | 62.5% | 69.1% | 73.3% | 1.52 | 1.32 | 18.9% | 88.5% | 18.5 | 1.82x | 1.66x |
| toolcall/train | corpus | 4 | 2500 | 47.7% | 56.2% | 62.2% | 2.71 | 2.24 | 19.2% | 88.5% | 27.3 | 2.33x | 1.98x |
| toolcall/train | corpus | 8 | 2500 | 28.9% | 38.9% | 48.1% | 4.42 | 3.33 | 19.2% | 88.5% | 34.8 | 2.86x | 2.40x |
| toolcall/train | corpus | 16 | 2500 | 16.1% | 26.5% | 31.4% | 6.29 | 4.43 | 19.6% | 88.5% | 42.2 | 3.16x | 2.56x |
| toolcall/train | schema | 2 | 2500 | 52.1% | 66.0% | 67.8% | 1.42 | 1.16 | 23.8% | 82.3% | 3.0 | 1.80x | 1.65x |
| toolcall/train | schema | 4 | 2500 | 40.1% | 53.5% | 56.0% | 2.51 | 1.95 | 23.8% | 82.3% | 4.7 | 2.31x | 1.96x |
| toolcall/train | schema | 8 | 2500 | 27.8% | 33.4% | 33.5% | 3.81 | 3.01 | 23.8% | 82.3% | 6.7 | 2.61x | 2.23x |
| toolcall/train | schema | 16 | 2500 | 2.2% | 3.5% | 3.8% | 4.80 | 3.75 | 23.8% | 82.3% | 7.4 | 2.84x | 2.40x |
| toolcall/train | finite | 2 | 2500 | 68.4% | 79.7% | 85.2% | 1.73 | 1.46 | 9.8% | 98.3% | 28.2 | 1.98x | 1.80x |
| toolcall/train | finite | 4 | 2500 | 51.9% | 64.0% | 76.2% | 3.16 | 2.45 | 10.0% | 98.3% | 39.6 | 2.74x | 2.21x |
| toolcall/train | finite | 8 | 2500 | 30.4% | 44.3% | 63.1% | 5.38 | 3.57 | 10.0% | 98.3% | 50.2 | 3.56x | 2.65x |
| toolcall/train | finite | 16 | 2500 | 16.1% | 26.5% | 33.1% | 7.09 | 4.61 | 10.2% | 98.3% | 58.8 | 4.01x | 2.86x |
| toolcall/train | dflash | 2 | 2500 | 81.4% | 82.1% | 82.1% | 1.73 | 1.69 | 6.5% | 100.0% | 4.0 | 1.97x | 1.92x |
| toolcall/train | dflash | 4 | 2500 | 72.3% | 72.5% | 72.5% | 3.11 | 3.07 | 6.5% | 100.0% | 4.0 | 2.64x | 2.57x |
| toolcall/train | dflash | 8 | 2500 | 46.1% | 46.2% | 46.2% | 4.98 | 4.93 | 6.5% | 100.0% | 4.0 | 3.23x | 3.12x |
| toolcall/train | dflash | 16 | 2500 | 0.0% | 0.0% | 0.0% | 6.04 | 5.99 | 6.5% | 100.0% | 4.0 | 3.33x | 3.22x |
| toolcall/train | union | 2 | 2500 | 81.4% | 82.1% | 94.0% | 1.88 | 1.69 | 3.3% | 100.0% | 11.7 | 2.13x | 1.92x |
| toolcall/train | union | 4 | 2500 | 72.3% | 72.5% | 89.2% | 3.54 | 3.07 | 3.3% | 100.0% | 12.8 | 3.13x | 2.57x |
| toolcall/train | union | 8 | 2500 | 46.1% | 46.2% | 77.7% | 6.26 | 4.93 | 3.3% | 100.0% | 13.4 | 4.44x | 3.12x |
| toolcall/train | union | 16 | 2500 | 0.0% | 0.0% | 31.1% | 8.62 | 5.99 | 3.3% | 100.0% | 13.9 | 5.30x | 3.22x |
