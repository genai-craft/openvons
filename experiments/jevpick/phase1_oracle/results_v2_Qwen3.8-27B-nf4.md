# Oracle Study v2: oracle_v2_Qwen3.8-27B-nf4

推定 speedup = replay (候補なし位置は通常 decode) × 実測 forward コスト比 (HF eager, ctx=1024)。

## G0 表 (L=4, oracle recall@16) と mean max match

| Group | ngram | grammar | corpus | schema | finite | dflash | union |
|---|---:|---:|---:|---:|---:|---:|---:|
| toolcall/test/known | 32.5% / 1.95 | 0.0% / 0.06 | 89.0% / 3.61 | 78.0% / 3.14 | 90.4% / 3.65 | 96.1% / 3.80 | 99.3% / 3.85 |
| toolcall/test/unknown | 30.2% / 1.85 | 0.0% / 0.05 | 57.2% / 2.67 | 75.9% / 3.06 | 76.2% / 3.27 | 92.0% / 3.72 | 95.3% / 3.78 |

## 推定 speedup (oracle / prior top-1), source × L

| Group | Source | L=2 | L=4 | L=8 | L=16 |
|---|---|---:|---:|---:|---:|
| toolcall/test/known | ngram | 1.72x / 1.47x | 2.23x / 2.05x | 2.39x / 1.66x | 2.40x / 1.62x |
| toolcall/test/known | grammar | 1.00x / 0.99x | 1.00x / 0.96x | 0.99x / 0.96x | 0.98x / 0.95x |
| toolcall/test/known | corpus | 2.22x / 2.06x | 3.48x / 2.88x | 5.39x / 4.11x | 7.31x / 5.75x |
| toolcall/test/known | schema | 1.95x / 1.78x | 2.73x / 2.33x | 3.43x / 2.93x | 3.80x / 2.89x |
| toolcall/test/known | finite | 2.25x / 2.07x | 3.56x / 2.91x | 5.61x / 4.21x | 7.70x / 5.90x |
| toolcall/test/known | dflash | 2.28x / 2.27x | 3.67x / 3.66x | 5.28x / 5.27x | 5.10x / 5.09x |
| toolcall/test/known | union | 2.28x / 2.27x | 3.70x / 3.66x | 5.91x / 5.27x | 8.11x / 5.09x |
| toolcall/test/unknown | ngram | 1.65x / 1.44x | 2.11x / 1.95x | 2.24x / 1.61x | 2.23x / 1.57x |
| toolcall/test/unknown | grammar | 0.99x / 0.99x | 0.99x / 0.96x | 0.99x / 0.96x | 0.98x / 0.95x |
| toolcall/test/unknown | corpus | 1.85x / 1.69x | 2.45x / 2.12x | 3.01x / 2.36x | 3.29x / 2.58x |
| toolcall/test/unknown | schema | 1.89x / 1.71x | 2.59x / 2.20x | 3.21x / 2.64x | 3.45x / 2.61x |
| toolcall/test/unknown | finite | 2.11x / 1.87x | 3.02x / 2.44x | 4.26x / 3.04x | 4.77x / 3.49x |
| toolcall/test/unknown | dflash | 2.27x / 2.26x | 3.55x / 3.53x | 4.99x / 4.94x | 4.82x / 4.77x |
| toolcall/test/unknown | union | 2.28x / 2.26x | 3.61x / 3.53x | 5.46x / 4.94x | 6.17x / 4.77x |

## 詳細

| Group | Source | L | n_pos | recall@1 | recall@4 | recall@16 | mean max@16 | mean top-1 | zero-hit@16 | 候補あり率 | 平均候補数 | speedup oracle | speedup top-1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toolcall/test/known | ngram | 2 | 10998 | 39.9% | 53.0% | 53.5% | 1.24 | 0.99 | 28.0% | 94.3% | 8.1 | 1.72x | 1.47x |
| toolcall/test/known | ngram | 4 | 10998 | 17.8% | 32.3% | 32.5% | 1.95 | 1.44 | 28.1% | 94.3% | 8.9 | 2.23x | 2.05x |
| toolcall/test/known | ngram | 8 | 10998 | 1.4% | 4.2% | 5.3% | 2.36 | 1.27 | 28.1% | 94.3% | 10.1 | 2.39x | 1.66x |
| toolcall/test/known | ngram | 16 | 10998 | 0.0% | 0.1% | 0.1% | 2.43 | 1.29 | 28.1% | 94.3% | 10.2 | 2.40x | 1.62x |
| toolcall/test/known | grammar | 2 | 10998 | 0.0% | 0.0% | 0.0% | 0.06 | 0.05 | 94.4% | 26.2% | 0.8 | 1.00x | 0.99x |
| toolcall/test/known | grammar | 4 | 10998 | 0.0% | 0.0% | 0.0% | 0.06 | 0.03 | 94.4% | 26.2% | 1.0 | 1.00x | 0.96x |
| toolcall/test/known | grammar | 8 | 10998 | 0.0% | 0.0% | 0.0% | 0.06 | 0.03 | 94.4% | 26.2% | 1.0 | 0.99x | 0.96x |
| toolcall/test/known | grammar | 16 | 10998 | 0.0% | 0.0% | 0.0% | 0.06 | 0.03 | 94.4% | 26.2% | 1.0 | 0.98x | 0.95x |
| toolcall/test/known | corpus | 2 | 10998 | 82.3% | 91.7% | 95.5% | 1.91 | 1.69 | 2.3% | 99.5% | 8.1 | 2.22x | 2.06x |
| toolcall/test/known | corpus | 4 | 10998 | 68.6% | 82.8% | 89.0% | 3.61 | 3.01 | 2.3% | 99.5% | 14.4 | 3.48x | 2.88x |
| toolcall/test/known | corpus | 8 | 10998 | 52.0% | 68.2% | 75.9% | 6.38 | 4.98 | 2.8% | 99.5% | 24.2 | 5.39x | 4.11x |
| toolcall/test/known | corpus | 16 | 10998 | 38.0% | 54.9% | 63.3% | 10.34 | 7.66 | 3.0% | 99.5% | 41.1 | 7.31x | 5.75x |
| toolcall/test/known | schema | 2 | 10998 | 64.5% | 80.1% | 82.2% | 1.65 | 1.39 | 15.3% | 89.7% | 4.2 | 1.95x | 1.78x |
| toolcall/test/known | schema | 4 | 10998 | 44.0% | 68.9% | 78.0% | 3.14 | 2.30 | 15.3% | 89.7% | 7.4 | 2.73x | 2.33x |
| toolcall/test/known | schema | 8 | 10998 | 17.5% | 24.7% | 44.6% | 5.20 | 3.16 | 15.3% | 89.7% | 14.5 | 3.43x | 2.93x |
| toolcall/test/known | schema | 16 | 10998 | 0.0% | 0.3% | 0.4% | 5.84 | 3.50 | 15.3% | 89.7% | 22.3 | 3.80x | 2.89x |
| toolcall/test/known | finite | 2 | 10998 | 82.7% | 92.7% | 96.6% | 1.93 | 1.70 | 1.5% | 99.9% | 21.1 | 2.25x | 2.07x |
| toolcall/test/known | finite | 4 | 10998 | 68.7% | 83.6% | 90.4% | 3.65 | 3.02 | 1.4% | 99.9% | 31.7 | 3.56x | 2.91x |
| toolcall/test/known | finite | 8 | 10998 | 51.9% | 68.5% | 77.4% | 6.49 | 5.00 | 1.8% | 99.9% | 49.8 | 5.61x | 4.21x |
| toolcall/test/known | finite | 16 | 10998 | 38.0% | 54.9% | 63.2% | 10.43 | 7.68 | 2.2% | 99.9% | 74.5 | 7.70x | 5.90x |
| toolcall/test/known | dflash | 2 | 10998 | 99.2% | 99.2% | 99.2% | 1.97 | 1.97 | 0.1% | 100.0% | 4.0 | 2.28x | 2.27x |
| toolcall/test/known | dflash | 4 | 10998 | 96.1% | 96.1% | 96.1% | 3.80 | 3.80 | 0.1% | 100.0% | 4.0 | 3.67x | 3.66x |
| toolcall/test/known | dflash | 8 | 10998 | 0.0% | 0.0% | 0.0% | 6.16 | 6.16 | 0.1% | 100.0% | 4.0 | 5.28x | 5.27x |
| toolcall/test/known | dflash | 16 | 10998 | 0.0% | 0.0% | 0.0% | 6.16 | 6.16 | 0.1% | 100.0% | 4.0 | 5.10x | 5.09x |
| toolcall/test/known | union | 2 | 10998 | 99.2% | 99.2% | 99.8% | 1.97 | 1.97 | 0.0% | 100.0% | 13.5 | 2.28x | 2.27x |
| toolcall/test/known | union | 4 | 10998 | 96.1% | 96.2% | 99.3% | 3.85 | 3.80 | 0.0% | 100.0% | 14.6 | 3.70x | 3.66x |
| toolcall/test/known | union | 8 | 10998 | 0.0% | 0.0% | 75.5% | 7.04 | 6.16 | 0.0% | 100.0% | 15.2 | 5.91x | 5.27x |
| toolcall/test/known | union | 16 | 10998 | 0.0% | 0.0% | 61.9% | 10.99 | 6.16 | 0.0% | 100.0% | 15.6 | 8.11x | 5.09x |
| toolcall/test/unknown | ngram | 2 | 11320 | 38.0% | 50.8% | 51.3% | 1.19 | 0.95 | 31.2% | 92.5% | 7.7 | 1.65x | 1.44x |
| toolcall/test/unknown | ngram | 4 | 11320 | 15.8% | 29.9% | 30.2% | 1.85 | 1.36 | 31.2% | 92.5% | 8.5 | 2.11x | 1.95x |
| toolcall/test/unknown | ngram | 8 | 11320 | 0.7% | 3.3% | 4.4% | 2.20 | 1.17 | 31.2% | 92.5% | 9.6 | 2.24x | 1.61x |
| toolcall/test/unknown | ngram | 16 | 11320 | 0.0% | 0.1% | 0.1% | 2.27 | 1.18 | 31.2% | 92.5% | 9.8 | 2.23x | 1.57x |
| toolcall/test/unknown | grammar | 2 | 11320 | 0.0% | 0.0% | 0.0% | 0.05 | 0.05 | 95.2% | 24.2% | 0.7 | 0.99x | 0.99x |
| toolcall/test/unknown | grammar | 4 | 11320 | 0.0% | 0.0% | 0.0% | 0.05 | 0.03 | 95.2% | 24.2% | 0.9 | 0.99x | 0.96x |
| toolcall/test/unknown | grammar | 8 | 11320 | 0.0% | 0.0% | 0.0% | 0.05 | 0.03 | 95.2% | 24.2% | 0.9 | 0.99x | 0.96x |
| toolcall/test/unknown | grammar | 16 | 11320 | 0.0% | 0.0% | 0.0% | 0.05 | 0.03 | 95.2% | 24.2% | 0.9 | 0.98x | 0.95x |
| toolcall/test/unknown | corpus | 2 | 11320 | 60.4% | 69.5% | 73.6% | 1.54 | 1.31 | 18.2% | 92.4% | 9.5 | 1.85x | 1.69x |
| toolcall/test/unknown | corpus | 4 | 11320 | 41.1% | 52.9% | 57.2% | 2.67 | 2.11 | 18.7% | 92.4% | 16.9 | 2.45x | 2.12x |
| toolcall/test/unknown | corpus | 8 | 11320 | 22.2% | 30.9% | 35.7% | 4.15 | 3.03 | 18.8% | 92.4% | 27.9 | 3.01x | 2.36x |
| toolcall/test/unknown | corpus | 16 | 11320 | 12.0% | 17.1% | 20.1% | 5.56 | 3.95 | 19.3% | 92.4% | 45.2 | 3.29x | 2.58x |
| toolcall/test/unknown | schema | 2 | 11320 | 61.5% | 78.2% | 80.3% | 1.61 | 1.34 | 17.7% | 87.0% | 4.1 | 1.89x | 1.71x |
| toolcall/test/unknown | schema | 4 | 11320 | 40.8% | 66.8% | 75.9% | 3.06 | 2.18 | 17.7% | 87.0% | 7.5 | 2.59x | 2.20x |
| toolcall/test/unknown | schema | 8 | 11320 | 15.9% | 24.8% | 42.3% | 5.03 | 2.95 | 17.7% | 87.0% | 15.6 | 3.21x | 2.64x |
| toolcall/test/unknown | schema | 16 | 11320 | 0.9% | 1.3% | 1.5% | 5.59 | 3.25 | 17.7% | 87.0% | 25.2 | 3.45x | 2.61x |
| toolcall/test/unknown | finite | 2 | 11320 | 70.4% | 82.1% | 88.4% | 1.80 | 1.51 | 6.9% | 99.0% | 22.1 | 2.11x | 1.87x |
| toolcall/test/unknown | finite | 4 | 11320 | 50.8% | 66.1% | 76.2% | 3.27 | 2.52 | 7.1% | 99.0% | 33.7 | 3.02x | 2.44x |
| toolcall/test/unknown | finite | 8 | 11320 | 24.4% | 36.3% | 46.8% | 5.24 | 3.65 | 7.4% | 99.0% | 54.0 | 4.26x | 3.04x |
| toolcall/test/unknown | finite | 16 | 11320 | 11.4% | 16.7% | 20.2% | 6.69 | 4.58 | 8.5% | 99.0% | 81.1 | 4.77x | 3.49x |
| toolcall/test/unknown | dflash | 2 | 11320 | 97.3% | 97.5% | 97.5% | 1.95 | 1.94 | 0.2% | 100.0% | 4.0 | 2.27x | 2.26x |
| toolcall/test/unknown | dflash | 4 | 11320 | 91.9% | 92.0% | 92.0% | 3.72 | 3.71 | 0.2% | 100.0% | 4.0 | 3.55x | 3.53x |
| toolcall/test/unknown | dflash | 8 | 11320 | 0.0% | 0.0% | 0.0% | 5.94 | 5.92 | 0.2% | 100.0% | 4.0 | 4.99x | 4.94x |
| toolcall/test/unknown | dflash | 16 | 11320 | 0.0% | 0.0% | 0.0% | 5.94 | 5.92 | 0.2% | 100.0% | 4.0 | 4.82x | 4.77x |
| toolcall/test/unknown | union | 2 | 11320 | 97.3% | 97.5% | 98.4% | 1.96 | 1.94 | 0.1% | 100.0% | 13.4 | 2.28x | 2.26x |
| toolcall/test/unknown | union | 4 | 11320 | 91.9% | 92.0% | 95.3% | 3.78 | 3.71 | 0.1% | 100.0% | 14.3 | 3.61x | 3.53x |
| toolcall/test/unknown | union | 8 | 11320 | 0.0% | 0.0% | 44.7% | 6.51 | 5.92 | 0.1% | 100.0% | 14.8 | 5.46x | 4.94x |
| toolcall/test/unknown | union | 16 | 11320 | 0.0% | 0.0% | 19.6% | 8.03 | 5.92 | 0.1% | 100.0% | 15.1 | 6.17x | 4.77x |
