# Oracle Study v2: oracle_v2_Qwen3.8-27B-nf4

推定 speedup = replay (候補なし位置は通常 decode) × 実測 forward コスト比 (HF eager, ctx=1024)。

## G0 表 (L=4, oracle recall@16) と mean max match

| Group | ngram | grammar | corpus | schema | finite | dflash | union |
|---|---:|---:|---:|---:|---:|---:|---:|
| toolcall/test/known | 32.9% / 1.94 | 0.0% / 0.05 | 93.1% / 3.66 | 87.3% / 3.41 | 98.3% / 3.81 | 97.4% / 3.81 | 100.0% / 3.84 |
| toolcall/test/unknown | 27.5% / 1.81 | 0.0% / 0.05 | 35.2% / 1.82 | 64.3% / 2.63 | 68.4% / 2.96 | 89.9% / 3.71 | 90.3% / 3.72 |
| toolcall/train | 31.7% / 1.91 | 0.0% / 0.06 | 59.1% / 2.68 | 80.4% / 3.21 | 89.6% / 3.59 | 96.7% / 3.81 | 99.2% / 3.83 |

## 推定 speedup (oracle / prior top-1), source × L

| Group | Source | L=2 | L=4 | L=8 | L=16 |
|---|---|---:|---:|---:|---:|
| toolcall/test/known | ngram | 1.71x / 1.43x | 2.25x / 2.06x | 2.38x / 1.58x | 2.41x / 1.55x |
| toolcall/test/known | grammar | 1.00x / 1.00x | 0.99x / 0.96x | 0.99x / 0.96x | 0.98x / 0.95x |
| toolcall/test/known | corpus | 2.23x / 2.04x | 3.42x / 3.06x | 5.23x / 4.00x | 6.47x / 4.80x |
| toolcall/test/known | schema | 2.06x / 1.90x | 3.07x / 2.65x | 4.19x / 3.61x | 4.83x / 3.89x |
| toolcall/test/known | finite | 2.27x / 1.89x | 3.60x / 2.73x | 5.57x / 3.62x | 7.56x / 4.30x |
| toolcall/test/known | dflash | 2.27x / 2.27x | 3.65x / 3.65x | 5.26x / 5.26x | 5.08x / 5.08x |
| toolcall/test/known | union | 2.27x / 2.27x | 3.65x / 3.65x | 5.91x / 5.26x | 7.99x / 5.08x |
| toolcall/test/unknown | ngram | 1.65x / 1.39x | 2.05x / 1.75x | 2.14x / 1.55x | 2.10x / 1.52x |
| toolcall/test/unknown | grammar | 1.00x / 0.99x | 1.00x / 0.97x | 1.00x / 0.96x | 0.99x / 0.95x |
| toolcall/test/unknown | corpus | 1.48x / 1.37x | 1.71x / 1.57x | 1.82x / 1.55x | 1.88x / 1.60x |
| toolcall/test/unknown | schema | 1.69x / 1.55x | 2.11x / 1.81x | 2.43x / 1.99x | 2.50x / 1.99x |
| toolcall/test/unknown | finite | 1.87x / 1.54x | 2.51x / 1.84x | 2.99x / 2.05x | 3.21x / 2.12x |
| toolcall/test/unknown | dflash | 2.27x / 2.27x | 3.62x / 3.62x | 5.18x / 5.13x | 5.00x / 4.95x |
| toolcall/test/unknown | union | 2.27x / 2.27x | 3.65x / 3.62x | 5.41x / 5.13x | 5.73x / 4.95x |
| toolcall/train | ngram | 1.71x / 1.47x | 2.24x / 2.09x | 2.36x / 1.66x | 2.39x / 1.61x |
| toolcall/train | grammar | 1.00x / 1.00x | 1.00x / 0.96x | 0.99x / 0.96x | 0.98x / 0.95x |
| toolcall/train | corpus | 1.82x / 1.67x | 2.42x / 2.19x | 2.85x / 2.34x | 3.16x / 2.51x |
| toolcall/train | schema | 1.97x / 1.82x | 2.84x / 2.50x | 3.67x / 3.17x | 4.13x / 3.14x |
| toolcall/train | finite | 2.18x / 1.85x | 3.34x / 2.58x | 4.66x / 3.09x | 5.50x / 3.32x |
| toolcall/train | dflash | 2.28x / 2.28x | 3.66x / 3.66x | 5.22x / 5.22x | 5.04x / 5.04x |
| toolcall/train | union | 2.28x / 2.28x | 3.66x / 3.66x | 5.75x / 5.22x | 6.55x / 5.04x |

## 詳細

| Group | Source | L | n_pos | recall@1 | recall@4 | recall@16 | mean max@16 | mean top-1 | zero-hit@16 | 候補あり率 | 平均候補数 | speedup oracle | speedup top-1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toolcall/test/known | ngram | 2 | 377 | 39.5% | 53.7% | 53.7% | 1.24 | 0.97 | 28.4% | 93.9% | 8.1 | 1.71x | 1.43x |
| toolcall/test/known | ngram | 4 | 377 | 17.3% | 32.9% | 32.9% | 1.94 | 1.41 | 28.4% | 93.9% | 8.8 | 2.25x | 2.06x |
| toolcall/test/known | ngram | 8 | 377 | 1.6% | 4.9% | 6.2% | 2.38 | 1.21 | 28.4% | 93.9% | 9.9 | 2.38x | 1.58x |
| toolcall/test/known | ngram | 16 | 377 | 0.0% | 0.0% | 0.0% | 2.42 | 1.24 | 28.4% | 93.9% | 10.0 | 2.41x | 1.55x |
| toolcall/test/known | grammar | 2 | 377 | 0.0% | 0.0% | 0.0% | 0.05 | 0.05 | 94.7% | 26.0% | 0.8 | 1.00x | 1.00x |
| toolcall/test/known | grammar | 4 | 377 | 0.0% | 0.0% | 0.0% | 0.05 | 0.03 | 94.7% | 26.0% | 1.0 | 0.99x | 0.96x |
| toolcall/test/known | grammar | 8 | 377 | 0.0% | 0.0% | 0.0% | 0.05 | 0.03 | 94.7% | 26.0% | 1.0 | 0.99x | 0.96x |
| toolcall/test/known | grammar | 16 | 377 | 0.0% | 0.0% | 0.0% | 0.05 | 0.03 | 94.7% | 26.0% | 1.0 | 0.98x | 0.95x |
| toolcall/test/known | corpus | 2 | 377 | 82.3% | 93.2% | 96.2% | 1.92 | 1.71 | 2.1% | 99.2% | 17.3 | 2.23x | 2.04x |
| toolcall/test/known | corpus | 4 | 377 | 67.1% | 87.0% | 93.1% | 3.66 | 3.01 | 2.1% | 99.2% | 31.1 | 3.42x | 3.06x |
| toolcall/test/known | corpus | 8 | 377 | 50.2% | 72.3% | 83.4% | 6.59 | 4.79 | 2.1% | 99.2% | 43.2 | 5.23x | 4.00x |
| toolcall/test/known | corpus | 16 | 377 | 38.8% | 57.7% | 69.2% | 10.56 | 7.36 | 2.4% | 99.2% | 63.8 | 6.47x | 4.80x |
| toolcall/test/known | schema | 2 | 377 | 71.7% | 87.7% | 89.1% | 1.77 | 1.52 | 10.1% | 93.1% | 4.1 | 2.06x | 1.90x |
| toolcall/test/known | schema | 4 | 377 | 53.3% | 78.4% | 87.3% | 3.41 | 2.62 | 10.1% | 93.1% | 7.0 | 3.07x | 2.65x |
| toolcall/test/known | schema | 8 | 377 | 21.8% | 31.3% | 56.0% | 5.83 | 3.60 | 10.1% | 93.1% | 12.7 | 4.19x | 3.61x |
| toolcall/test/known | schema | 16 | 377 | 0.0% | 0.0% | 0.0% | 6.83 | 4.12 | 10.1% | 93.1% | 17.9 | 4.83x | 3.89x |
| toolcall/test/known | finite | 2 | 377 | 77.9% | 97.0% | 99.5% | 1.97 | 1.64 | 0.3% | 100.0% | 30.2 | 2.27x | 1.89x |
| toolcall/test/known | finite | 4 | 377 | 58.2% | 89.6% | 98.3% | 3.81 | 2.81 | 0.3% | 100.0% | 47.9 | 3.60x | 2.73x |
| toolcall/test/known | finite | 8 | 377 | 44.0% | 72.6% | 89.9% | 6.97 | 4.57 | 0.3% | 100.0% | 66.8 | 5.57x | 3.62x |
| toolcall/test/known | finite | 16 | 377 | 38.8% | 55.5% | 69.2% | 11.34 | 7.31 | 0.5% | 100.0% | 92.6 | 7.56x | 4.30x |
| toolcall/test/known | dflash | 2 | 377 | 100.0% | 100.0% | 100.0% | 1.97 | 1.97 | 0.0% | 100.0% | 4.0 | 2.27x | 2.27x |
| toolcall/test/known | dflash | 4 | 377 | 97.4% | 97.4% | 97.4% | 3.81 | 3.81 | 0.0% | 100.0% | 4.0 | 3.65x | 3.65x |
| toolcall/test/known | dflash | 8 | 377 | 0.0% | 0.0% | 0.0% | 6.13 | 6.13 | 0.0% | 100.0% | 4.0 | 5.26x | 5.26x |
| toolcall/test/known | dflash | 16 | 377 | 0.0% | 0.0% | 0.0% | 6.13 | 6.13 | 0.0% | 100.0% | 4.0 | 5.08x | 5.08x |
| toolcall/test/known | union | 2 | 377 | 100.0% | 100.0% | 100.0% | 1.97 | 1.97 | 0.0% | 100.0% | 13.4 | 2.27x | 2.27x |
| toolcall/test/known | union | 4 | 377 | 97.4% | 97.4% | 100.0% | 3.84 | 3.81 | 0.0% | 100.0% | 14.2 | 3.65x | 3.65x |
| toolcall/test/known | union | 8 | 377 | 0.0% | 0.0% | 87.9% | 7.11 | 6.13 | 0.0% | 100.0% | 14.7 | 5.91x | 5.26x |
| toolcall/test/known | union | 16 | 377 | 0.0% | 0.0% | 65.6% | 11.34 | 6.13 | 0.0% | 100.0% | 14.8 | 7.99x | 5.08x |
| toolcall/test/unknown | ngram | 2 | 641 | 35.8% | 49.3% | 50.2% | 1.19 | 0.90 | 30.7% | 92.5% | 7.4 | 1.65x | 1.39x |
| toolcall/test/unknown | ngram | 4 | 641 | 13.3% | 26.8% | 27.5% | 1.81 | 1.23 | 30.6% | 92.5% | 8.3 | 2.05x | 1.75x |
| toolcall/test/unknown | ngram | 8 | 641 | 0.5% | 2.3% | 2.8% | 2.09 | 1.08 | 30.9% | 92.5% | 9.5 | 2.14x | 1.55x |
| toolcall/test/unknown | ngram | 16 | 641 | 0.0% | 0.0% | 0.0% | 2.12 | 1.09 | 30.9% | 92.5% | 9.8 | 2.10x | 1.52x |
| toolcall/test/unknown | grammar | 2 | 641 | 0.0% | 0.0% | 0.0% | 0.05 | 0.05 | 94.5% | 23.7% | 0.8 | 1.00x | 0.99x |
| toolcall/test/unknown | grammar | 4 | 641 | 0.0% | 0.0% | 0.0% | 0.05 | 0.03 | 94.5% | 23.7% | 1.0 | 1.00x | 0.97x |
| toolcall/test/unknown | grammar | 8 | 641 | 0.0% | 0.0% | 0.0% | 0.05 | 0.02 | 94.5% | 23.7% | 1.0 | 1.00x | 0.96x |
| toolcall/test/unknown | grammar | 16 | 641 | 0.0% | 0.0% | 0.0% | 0.05 | 0.02 | 94.5% | 23.7% | 1.0 | 0.99x | 0.95x |
| toolcall/test/unknown | corpus | 2 | 641 | 37.6% | 47.1% | 49.8% | 1.09 | 0.87 | 40.4% | 75.2% | 13.0 | 1.48x | 1.37x |
| toolcall/test/unknown | corpus | 4 | 641 | 23.9% | 33.1% | 35.2% | 1.82 | 1.33 | 40.7% | 75.2% | 22.5 | 1.71x | 1.57x |
| toolcall/test/unknown | corpus | 8 | 641 | 9.8% | 12.6% | 15.9% | 2.60 | 1.83 | 40.7% | 75.2% | 30.9 | 1.82x | 1.55x |
| toolcall/test/unknown | corpus | 16 | 641 | 0.2% | 0.6% | 2.2% | 2.98 | 1.93 | 40.7% | 75.2% | 45.3 | 1.88x | 1.60x |
| toolcall/test/unknown | schema | 2 | 641 | 50.4% | 66.4% | 67.8% | 1.38 | 1.12 | 29.2% | 76.1% | 3.5 | 1.69x | 1.55x |
| toolcall/test/unknown | schema | 4 | 641 | 30.9% | 57.6% | 64.3% | 2.63 | 1.75 | 29.2% | 76.1% | 6.3 | 2.11x | 1.81x |
| toolcall/test/unknown | schema | 8 | 641 | 11.4% | 20.5% | 38.2% | 4.43 | 2.29 | 29.2% | 76.1% | 14.1 | 2.43x | 1.99x |
| toolcall/test/unknown | schema | 16 | 641 | 0.4% | 2.9% | 3.1% | 4.82 | 2.44 | 29.2% | 76.1% | 25.0 | 2.50x | 1.99x |
| toolcall/test/unknown | finite | 2 | 641 | 53.9% | 72.4% | 79.7% | 1.65 | 1.21 | 13.9% | 96.4% | 24.6 | 1.87x | 1.54x |
| toolcall/test/unknown | finite | 4 | 641 | 32.7% | 54.0% | 68.4% | 2.96 | 1.89 | 15.3% | 96.4% | 38.0 | 2.51x | 1.84x |
| toolcall/test/unknown | finite | 8 | 641 | 11.2% | 20.7% | 39.1% | 4.72 | 2.62 | 15.8% | 96.4% | 55.4 | 2.99x | 2.05x |
| toolcall/test/unknown | finite | 16 | 641 | 0.2% | 0.4% | 4.9% | 5.71 | 2.75 | 15.8% | 96.4% | 81.1 | 3.21x | 2.12x |
| toolcall/test/unknown | dflash | 2 | 641 | 96.2% | 96.4% | 96.4% | 1.95 | 1.93 | 0.3% | 100.0% | 4.0 | 2.27x | 2.27x |
| toolcall/test/unknown | dflash | 4 | 641 | 89.7% | 89.9% | 89.9% | 3.71 | 3.69 | 0.3% | 100.0% | 4.0 | 3.62x | 3.62x |
| toolcall/test/unknown | dflash | 8 | 641 | 0.0% | 0.0% | 0.0% | 5.97 | 5.95 | 0.3% | 100.0% | 4.0 | 5.18x | 5.13x |
| toolcall/test/unknown | dflash | 16 | 641 | 0.0% | 0.0% | 0.0% | 5.97 | 5.95 | 0.3% | 100.0% | 4.0 | 5.00x | 4.95x |
| toolcall/test/unknown | union | 2 | 641 | 96.2% | 96.4% | 96.5% | 1.95 | 1.93 | 0.3% | 100.0% | 12.5 | 2.27x | 2.27x |
| toolcall/test/unknown | union | 4 | 641 | 89.7% | 89.9% | 90.3% | 3.72 | 3.69 | 0.3% | 100.0% | 13.2 | 3.65x | 3.62x |
| toolcall/test/unknown | union | 8 | 641 | 0.0% | 0.0% | 38.4% | 6.36 | 5.95 | 0.3% | 100.0% | 13.5 | 5.41x | 5.13x |
| toolcall/test/unknown | union | 16 | 641 | 0.0% | 0.0% | 4.9% | 7.34 | 5.95 | 0.3% | 100.0% | 13.6 | 5.73x | 4.95x |
| toolcall/train | ngram | 2 | 1617 | 39.1% | 52.6% | 52.7% | 1.23 | 0.98 | 28.8% | 93.6% | 8.2 | 1.71x | 1.47x |
| toolcall/train | ngram | 4 | 1617 | 18.0% | 31.7% | 31.7% | 1.91 | 1.44 | 28.8% | 93.6% | 8.9 | 2.24x | 2.09x |
| toolcall/train | ngram | 8 | 1617 | 0.6% | 3.3% | 5.0% | 2.31 | 1.24 | 28.8% | 93.6% | 10.0 | 2.36x | 1.66x |
| toolcall/train | ngram | 16 | 1617 | 0.0% | 0.0% | 0.0% | 2.35 | 1.24 | 28.8% | 93.6% | 10.1 | 2.39x | 1.61x |
| toolcall/train | grammar | 2 | 1617 | 0.0% | 0.0% | 0.0% | 0.06 | 0.05 | 94.4% | 26.2% | 0.8 | 1.00x | 1.00x |
| toolcall/train | grammar | 4 | 1617 | 0.0% | 0.0% | 0.0% | 0.06 | 0.03 | 94.4% | 26.2% | 1.0 | 1.00x | 0.96x |
| toolcall/train | grammar | 8 | 1617 | 0.0% | 0.0% | 0.0% | 0.06 | 0.03 | 94.4% | 26.2% | 1.0 | 0.99x | 0.96x |
| toolcall/train | grammar | 16 | 1617 | 0.0% | 0.0% | 0.0% | 0.06 | 0.03 | 94.4% | 26.2% | 1.0 | 0.98x | 0.95x |
| toolcall/train | corpus | 2 | 1617 | 59.9% | 68.9% | 73.0% | 1.51 | 1.30 | 19.9% | 87.5% | 13.3 | 1.82x | 1.67x |
| toolcall/train | corpus | 4 | 1617 | 42.1% | 53.3% | 59.1% | 2.68 | 2.09 | 20.0% | 87.5% | 24.4 | 2.42x | 2.19x |
| toolcall/train | corpus | 8 | 1617 | 26.6% | 32.9% | 39.3% | 4.24 | 3.16 | 20.0% | 87.5% | 34.0 | 2.85x | 2.34x |
| toolcall/train | corpus | 16 | 1617 | 15.4% | 18.1% | 23.8% | 5.77 | 4.16 | 20.5% | 87.5% | 49.1 | 3.16x | 2.51x |
| toolcall/train | schema | 2 | 1617 | 67.7% | 82.4% | 84.3% | 1.69 | 1.45 | 13.7% | 90.6% | 4.1 | 1.97x | 1.82x |
| toolcall/train | schema | 4 | 1617 | 48.0% | 71.7% | 80.4% | 3.21 | 2.45 | 13.7% | 90.6% | 7.3 | 2.84x | 2.50x |
| toolcall/train | schema | 8 | 1617 | 20.0% | 27.1% | 48.5% | 5.37 | 3.38 | 13.7% | 90.6% | 14.1 | 3.67x | 3.17x |
| toolcall/train | schema | 16 | 1617 | 0.0% | 0.0% | 0.1% | 6.14 | 3.75 | 13.7% | 90.6% | 20.6 | 4.13x | 3.14x |
| toolcall/train | finite | 2 | 1617 | 71.5% | 91.8% | 94.5% | 1.89 | 1.54 | 3.6% | 98.6% | 26.4 | 2.18x | 1.85x |
| toolcall/train | finite | 4 | 1617 | 48.8% | 79.0% | 89.6% | 3.59 | 2.51 | 3.6% | 98.6% | 41.5 | 3.34x | 2.58x |
| toolcall/train | finite | 8 | 1617 | 23.3% | 43.5% | 61.0% | 6.07 | 3.71 | 3.6% | 98.6% | 59.1 | 4.66x | 3.09x |
| toolcall/train | finite | 16 | 1617 | 12.8% | 17.4% | 22.8% | 8.05 | 4.49 | 3.6% | 98.6% | 80.9 | 5.50x | 3.32x |
| toolcall/train | dflash | 2 | 1617 | 99.7% | 99.7% | 99.7% | 1.97 | 1.97 | 0.1% | 100.0% | 4.0 | 2.28x | 2.28x |
| toolcall/train | dflash | 4 | 1617 | 96.7% | 96.7% | 96.7% | 3.81 | 3.81 | 0.1% | 100.0% | 4.0 | 3.66x | 3.66x |
| toolcall/train | dflash | 8 | 1617 | 0.0% | 0.0% | 0.0% | 6.14 | 6.14 | 0.1% | 100.0% | 4.0 | 5.22x | 5.22x |
| toolcall/train | dflash | 16 | 1617 | 0.0% | 0.0% | 0.0% | 6.14 | 6.14 | 0.1% | 100.0% | 4.0 | 5.04x | 5.04x |
| toolcall/train | union | 2 | 1617 | 99.7% | 99.7% | 99.7% | 1.97 | 1.97 | 0.1% | 100.0% | 13.2 | 2.28x | 2.28x |
| toolcall/train | union | 4 | 1617 | 96.7% | 96.7% | 99.2% | 3.83 | 3.81 | 0.1% | 100.0% | 13.9 | 3.66x | 3.66x |
| toolcall/train | union | 8 | 1617 | 0.0% | 0.0% | 60.1% | 6.84 | 6.14 | 0.1% | 100.0% | 14.3 | 5.75x | 5.22x |
| toolcall/train | union | 16 | 1617 | 0.0% | 0.0% | 22.1% | 8.84 | 6.14 | 0.1% | 100.0% | 14.5 | 6.55x | 5.04x |
