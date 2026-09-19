# Oracle Study v2: oracle_v2_Qwen3.8-27B-fp8

推定 speedup = replay (候補なし位置は通常 decode) × 実測 forward コスト比 (HF eager, ctx=1024)。

## G0 表 (L=4, oracle recall@16) と mean max match

| Group | ngram | grammar | corpus | schema | finite | dflash | union |
|---|---:|---:|---:|---:|---:|---:|---:|
| toolcall/test/known | 32.8% / 1.97 | 0.0% / 0.06 | 89.5% / 3.63 | 79.1% / 3.18 | 90.8% / 3.66 | 96.3% / 3.80 | 99.5% / 3.86 |
| toolcall/test/unknown | 32.1% / 1.95 | 0.0% / 0.05 | 59.5% / 2.76 | 80.5% / 3.24 | 79.7% / 3.38 | 95.4% / 3.79 | 98.0% / 3.83 |

## 推定 speedup (oracle / prior top-1), source × L

| Group | Source | L=2 | L=4 | L=8 | L=16 |
|---|---|---:|---:|---:|---:|
| toolcall/test/known | ngram | 1.73x / 1.48x | 2.26x / 2.08x | 2.42x / 1.68x | 2.44x / 1.64x |
| toolcall/test/known | grammar | 1.00x / 0.99x | 1.00x / 0.96x | 0.99x / 0.96x | 0.98x / 0.95x |
| toolcall/test/known | corpus | 2.23x / 2.07x | 3.51x / 2.92x | 5.49x / 4.19x | 7.51x / 5.96x |
| toolcall/test/known | schema | 1.97x / 1.80x | 2.79x / 2.37x | 3.54x / 3.01x | 3.94x / 2.97x |
| toolcall/test/known | finite | 2.25x / 2.08x | 3.58x / 2.94x | 5.68x / 4.27x | 7.85x / 6.07x |
| toolcall/test/known | dflash | 2.28x / 2.28x | 3.67x / 3.67x | 5.32x / 5.31x | 5.14x / 5.13x |
| toolcall/test/known | union | 2.28x / 2.28x | 3.70x / 3.67x | 5.96x / 5.31x | 8.19x / 5.13x |
| toolcall/test/unknown | ngram | 1.72x / 1.48x | 2.25x / 2.07x | 2.41x / 1.68x | 2.41x / 1.63x |
| toolcall/test/unknown | grammar | 0.99x / 0.99x | 0.99x / 0.96x | 0.99x / 0.96x | 0.98x / 0.95x |
| toolcall/test/unknown | corpus | 1.90x / 1.75x | 2.57x / 2.24x | 3.24x / 2.51x | 3.61x / 2.79x |
| toolcall/test/unknown | schema | 1.99x / 1.79x | 2.85x / 2.37x | 3.69x / 2.93x | 4.04x / 2.89x |
| toolcall/test/unknown | finite | 2.19x / 1.95x | 3.23x / 2.61x | 4.79x / 3.35x | 5.53x / 3.96x |
| toolcall/test/unknown | dflash | 2.28x / 2.28x | 3.65x / 3.64x | 5.21x / 5.20x | 5.03x / 5.02x |
| toolcall/test/unknown | union | 2.29x / 2.28x | 3.68x / 3.64x | 5.71x / 5.20x | 6.54x / 5.02x |

## 詳細

| Group | Source | L | n_pos | recall@1 | recall@4 | recall@16 | mean max@16 | mean top-1 | zero-hit@16 | 候補あり率 | 平均候補数 | speedup oracle | speedup top-1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toolcall/test/known | ngram | 2 | 10853 | 40.3% | 53.6% | 54.1% | 1.25 | 1.00 | 27.4% | 94.8% | 8.2 | 1.73x | 1.48x |
| toolcall/test/known | ngram | 4 | 10853 | 18.0% | 32.6% | 32.8% | 1.97 | 1.45 | 27.4% | 94.8% | 9.0 | 2.26x | 2.08x |
| toolcall/test/known | ngram | 8 | 10853 | 1.4% | 4.2% | 5.3% | 2.38 | 1.27 | 27.4% | 94.8% | 10.2 | 2.42x | 1.68x |
| toolcall/test/known | ngram | 16 | 10853 | 0.0% | 0.1% | 0.1% | 2.44 | 1.30 | 27.4% | 94.8% | 10.3 | 2.44x | 1.64x |
| toolcall/test/known | grammar | 2 | 10853 | 0.0% | 0.0% | 0.0% | 0.06 | 0.05 | 94.4% | 26.4% | 0.8 | 1.00x | 0.99x |
| toolcall/test/known | grammar | 4 | 10853 | 0.0% | 0.0% | 0.0% | 0.06 | 0.03 | 94.4% | 26.4% | 1.0 | 1.00x | 0.96x |
| toolcall/test/known | grammar | 8 | 10853 | 0.0% | 0.0% | 0.0% | 0.06 | 0.03 | 94.4% | 26.4% | 1.0 | 0.99x | 0.96x |
| toolcall/test/known | grammar | 16 | 10853 | 0.0% | 0.0% | 0.0% | 0.06 | 0.03 | 94.4% | 26.4% | 1.0 | 0.98x | 0.95x |
| toolcall/test/known | corpus | 2 | 10853 | 82.8% | 92.1% | 96.0% | 1.92 | 1.70 | 2.0% | 99.6% | 8.0 | 2.23x | 2.07x |
| toolcall/test/known | corpus | 4 | 10853 | 69.2% | 83.3% | 89.5% | 3.63 | 3.03 | 2.0% | 99.6% | 14.4 | 3.51x | 2.92x |
| toolcall/test/known | corpus | 8 | 10853 | 52.8% | 69.0% | 76.7% | 6.41 | 5.02 | 2.5% | 99.6% | 24.3 | 5.49x | 4.19x |
| toolcall/test/known | corpus | 16 | 10853 | 38.6% | 55.8% | 64.5% | 10.41 | 7.73 | 2.7% | 99.6% | 41.3 | 7.51x | 5.96x |
| toolcall/test/known | schema | 2 | 10853 | 65.3% | 81.1% | 83.3% | 1.67 | 1.41 | 14.2% | 90.7% | 4.2 | 1.97x | 1.80x |
| toolcall/test/known | schema | 4 | 10853 | 44.6% | 69.8% | 79.1% | 3.18 | 2.34 | 14.2% | 90.7% | 7.5 | 2.79x | 2.37x |
| toolcall/test/known | schema | 8 | 10853 | 17.8% | 25.1% | 45.2% | 5.27 | 3.20 | 14.2% | 90.7% | 14.7 | 3.54x | 3.01x |
| toolcall/test/known | schema | 16 | 10853 | 0.0% | 0.3% | 0.4% | 5.92 | 3.55 | 14.2% | 90.7% | 22.6 | 3.94x | 2.97x |
| toolcall/test/known | finite | 2 | 10853 | 83.2% | 93.0% | 96.9% | 1.93 | 1.71 | 1.3% | 99.9% | 21.2 | 2.25x | 2.08x |
| toolcall/test/known | finite | 4 | 10853 | 69.3% | 84.0% | 90.8% | 3.66 | 3.04 | 1.2% | 99.9% | 31.9 | 3.58x | 2.94x |
| toolcall/test/known | finite | 8 | 10853 | 52.6% | 69.1% | 78.0% | 6.51 | 5.04 | 1.5% | 99.9% | 50.2 | 5.68x | 4.27x |
| toolcall/test/known | finite | 16 | 10853 | 38.6% | 55.8% | 64.4% | 10.48 | 7.74 | 2.0% | 99.9% | 75.2 | 7.85x | 6.07x |
| toolcall/test/known | dflash | 2 | 10853 | 99.3% | 99.3% | 99.3% | 1.97 | 1.97 | 0.0% | 100.0% | 4.0 | 2.28x | 2.28x |
| toolcall/test/known | dflash | 4 | 10853 | 96.2% | 96.3% | 96.3% | 3.80 | 3.80 | 0.0% | 100.0% | 4.0 | 3.67x | 3.67x |
| toolcall/test/known | dflash | 8 | 10853 | 0.0% | 0.0% | 0.0% | 6.16 | 6.16 | 0.0% | 100.0% | 4.0 | 5.32x | 5.31x |
| toolcall/test/known | dflash | 16 | 10853 | 0.0% | 0.0% | 0.0% | 6.16 | 6.16 | 0.0% | 100.0% | 4.0 | 5.14x | 5.13x |
| toolcall/test/known | union | 2 | 10853 | 99.3% | 99.4% | 100.0% | 1.98 | 1.97 | 0.0% | 100.0% | 13.6 | 2.28x | 2.28x |
| toolcall/test/known | union | 4 | 10853 | 96.2% | 96.3% | 99.5% | 3.86 | 3.80 | 0.0% | 100.0% | 14.7 | 3.70x | 3.67x |
| toolcall/test/known | union | 8 | 10853 | 0.0% | 0.0% | 76.1% | 7.05 | 6.16 | 0.0% | 100.0% | 15.2 | 5.96x | 5.31x |
| toolcall/test/known | union | 16 | 10853 | 0.0% | 0.0% | 63.1% | 11.04 | 6.16 | 0.0% | 100.0% | 15.6 | 8.19x | 5.13x |
| toolcall/test/unknown | ngram | 2 | 10836 | 40.1% | 53.6% | 54.1% | 1.25 | 0.99 | 28.1% | 94.3% | 8.1 | 1.72x | 1.48x |
| toolcall/test/unknown | ngram | 4 | 10836 | 16.7% | 31.8% | 32.1% | 1.95 | 1.43 | 28.2% | 94.3% | 8.9 | 2.25x | 2.07x |
| toolcall/test/unknown | ngram | 8 | 10836 | 0.8% | 3.6% | 4.7% | 2.32 | 1.23 | 28.2% | 94.3% | 10.1 | 2.41x | 1.68x |
| toolcall/test/unknown | ngram | 16 | 10836 | 0.0% | 0.1% | 0.1% | 2.39 | 1.24 | 28.2% | 94.3% | 10.3 | 2.41x | 1.63x |
| toolcall/test/unknown | grammar | 2 | 10836 | 0.0% | 0.0% | 0.0% | 0.05 | 0.05 | 94.9% | 25.4% | 0.7 | 0.99x | 0.99x |
| toolcall/test/unknown | grammar | 4 | 10836 | 0.0% | 0.0% | 0.0% | 0.05 | 0.03 | 94.9% | 25.4% | 0.9 | 0.99x | 0.96x |
| toolcall/test/unknown | grammar | 8 | 10836 | 0.0% | 0.0% | 0.0% | 0.05 | 0.03 | 94.9% | 25.4% | 0.9 | 0.99x | 0.96x |
| toolcall/test/unknown | grammar | 16 | 10836 | 0.0% | 0.0% | 0.0% | 0.05 | 0.03 | 94.9% | 25.4% | 0.9 | 0.98x | 0.95x |
| toolcall/test/unknown | corpus | 2 | 10836 | 62.5% | 71.8% | 76.0% | 1.58 | 1.36 | 16.1% | 93.0% | 9.6 | 1.90x | 1.75x |
| toolcall/test/unknown | corpus | 4 | 10836 | 42.8% | 54.9% | 59.5% | 2.76 | 2.19 | 16.6% | 93.0% | 17.2 | 2.57x | 2.24x |
| toolcall/test/unknown | corpus | 8 | 10836 | 23.4% | 32.7% | 37.6% | 4.30 | 3.14 | 16.7% | 93.0% | 28.7 | 3.24x | 2.51x |
| toolcall/test/unknown | corpus | 16 | 10836 | 12.9% | 18.4% | 21.6% | 5.77 | 4.11 | 17.2% | 93.0% | 46.8 | 3.61x | 2.79x |
| toolcall/test/unknown | schema | 2 | 10836 | 65.2% | 82.8% | 85.0% | 1.70 | 1.41 | 12.9% | 91.7% | 4.3 | 1.99x | 1.79x |
| toolcall/test/unknown | schema | 4 | 10836 | 43.4% | 71.0% | 80.5% | 3.24 | 2.31 | 12.9% | 91.7% | 7.9 | 2.85x | 2.37x |
| toolcall/test/unknown | schema | 8 | 10836 | 17.1% | 26.6% | 45.2% | 5.33 | 3.13 | 12.9% | 91.7% | 16.6 | 3.69x | 2.93x |
| toolcall/test/unknown | schema | 16 | 10836 | 1.0% | 1.4% | 1.7% | 5.93 | 3.46 | 12.9% | 91.7% | 26.8 | 4.04x | 2.89x |
| toolcall/test/unknown | finite | 2 | 10836 | 73.1% | 85.0% | 91.5% | 1.85 | 1.56 | 4.5% | 99.4% | 22.7 | 2.19x | 1.95x |
| toolcall/test/unknown | finite | 4 | 10836 | 53.2% | 69.0% | 79.7% | 3.38 | 2.61 | 4.7% | 99.4% | 34.9 | 3.23x | 2.61x |
| toolcall/test/unknown | finite | 8 | 10836 | 25.8% | 38.6% | 49.6% | 5.45 | 3.80 | 5.0% | 99.4% | 56.3 | 4.79x | 3.35x |
| toolcall/test/unknown | finite | 16 | 10836 | 12.3% | 18.0% | 21.8% | 6.98 | 4.78 | 6.1% | 99.4% | 84.8 | 5.53x | 3.96x |
| toolcall/test/unknown | dflash | 2 | 10836 | 99.1% | 99.1% | 99.1% | 1.97 | 1.96 | 0.1% | 100.0% | 4.0 | 2.28x | 2.28x |
| toolcall/test/unknown | dflash | 4 | 10836 | 95.3% | 95.4% | 95.4% | 3.79 | 3.79 | 0.1% | 100.0% | 4.0 | 3.65x | 3.64x |
| toolcall/test/unknown | dflash | 8 | 10836 | 0.0% | 0.0% | 0.0% | 6.09 | 6.09 | 0.1% | 100.0% | 4.0 | 5.21x | 5.20x |
| toolcall/test/unknown | dflash | 16 | 10836 | 0.0% | 0.0% | 0.0% | 6.09 | 6.09 | 0.1% | 100.0% | 4.0 | 5.03x | 5.02x |
| toolcall/test/unknown | union | 2 | 10836 | 99.1% | 99.2% | 99.6% | 1.97 | 1.96 | 0.1% | 100.0% | 13.5 | 2.29x | 2.28x |
| toolcall/test/unknown | union | 4 | 10836 | 95.3% | 95.4% | 98.0% | 3.83 | 3.79 | 0.1% | 100.0% | 14.5 | 3.68x | 3.64x |
| toolcall/test/unknown | union | 8 | 10836 | 0.0% | 0.0% | 47.3% | 6.65 | 6.09 | 0.1% | 100.0% | 15.0 | 5.71x | 5.20x |
| toolcall/test/unknown | union | 16 | 10836 | 0.0% | 0.0% | 21.1% | 8.26 | 6.09 | 0.1% | 100.0% | 15.3 | 6.54x | 5.02x |
