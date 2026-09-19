# Oracle Study v2: oracle_v2_Qwen3-4B

推定 speedup = replay (候補なし位置は通常 decode) × 実測 forward コスト比 (HF eager, ctx=1024)。

## G0 表 (L=4, oracle recall@16) と mean max match

| Group | ngram | grammar | corpus | schema | finite | dflash | union |
|---|---:|---:|---:|---:|---:|---:|---:|
| toolcall/test/known | 21.5% / 1.81 | 10.8% / 0.89 | 68.2% / 2.92 | 57.5% / 2.59 | 82.0% / 3.35 | 77.6% / 3.26 | 92.8% / 3.61 |
| toolcall/test/unknown | 18.8% / 1.60 | 8.8% / 0.75 | 27.2% / 1.62 | 51.1% / 2.37 | 58.2% / 2.68 | 72.1% / 3.15 | 81.2% / 3.40 |
| toolcall/train | 21.4% / 1.78 | 10.4% / 0.87 | 63.7% / 2.77 | 58.1% / 2.59 | 77.8% / 3.21 | 76.9% / 3.21 | 89.9% / 3.54 |

## 推定 speedup (oracle / prior top-1), source × L

| Group | Source | L=2 | L=4 | L=8 | L=16 |
|---|---|---:|---:|---:|---:|
| toolcall/test/known | ngram | 1.63x / 1.33x | 1.94x / 1.52x | 1.94x / 1.55x | 1.90x / 1.50x |
| toolcall/test/known | grammar | 1.38x / 1.25x | 1.45x / 1.31x | 1.54x / 1.25x | 1.52x / 1.23x |
| toolcall/test/known | corpus | 1.92x / 1.75x | 2.62x / 2.18x | 3.23x / 2.67x | 3.77x / 2.91x |
| toolcall/test/known | schema | 1.87x / 1.66x | 2.43x / 1.97x | 2.80x / 2.27x | 3.11x / 2.43x |
| toolcall/test/known | finite | 2.07x / 1.82x | 3.01x / 2.30x | 4.03x / 3.03x | 4.89x / 3.26x |
| toolcall/test/known | dflash | 1.97x / 1.96x | 2.65x / 2.64x | 3.34x / 3.29x | 3.39x / 3.36x |
| toolcall/test/known | union | 2.13x / 1.96x | 3.25x / 2.64x | 4.51x / 3.29x | 5.58x / 3.36x |
| toolcall/test/unknown | ngram | 1.54x / 1.28x | 1.74x / 1.40x | 1.79x / 1.44x | 1.74x / 1.40x |
| toolcall/test/unknown | grammar | 1.30x / 1.20x | 1.36x / 1.25x | 1.43x / 1.21x | 1.41x / 1.19x |
| toolcall/test/unknown | corpus | 1.50x / 1.33x | 1.69x / 1.46x | 1.84x / 1.52x | 1.81x / 1.50x |
| toolcall/test/unknown | schema | 1.74x / 1.57x | 2.12x / 1.83x | 2.43x / 2.06x | 2.59x / 2.15x |
| toolcall/test/unknown | finite | 1.84x / 1.57x | 2.35x / 1.84x | 2.78x / 2.10x | 2.93x / 2.15x |
| toolcall/test/unknown | dflash | 2.01x / 1.98x | 2.73x / 2.69x | 3.40x / 3.29x | 3.46x / 3.35x |
| toolcall/test/unknown | union | 2.10x / 1.98x | 2.99x / 2.69x | 3.94x / 3.29x | 4.40x / 3.35x |
| toolcall/train | ngram | 1.62x / 1.33x | 1.87x / 1.49x | 1.89x / 1.52x | 1.85x / 1.48x |
| toolcall/train | grammar | 1.37x / 1.25x | 1.44x / 1.31x | 1.52x / 1.24x | 1.50x / 1.22x |
| toolcall/train | corpus | 1.85x / 1.71x | 2.40x / 2.03x | 2.98x / 2.47x | 3.32x / 2.72x |
| toolcall/train | schema | 1.85x / 1.68x | 2.41x / 2.01x | 2.74x / 2.32x | 3.01x / 2.51x |
| toolcall/train | finite | 2.01x / 1.83x | 2.82x / 2.26x | 3.71x / 2.74x | 4.22x / 2.98x |
| toolcall/train | dflash | 1.98x / 1.94x | 2.66x / 2.62x | 3.29x / 3.23x | 3.45x / 3.41x |
| toolcall/train | union | 2.12x / 1.94x | 3.13x / 2.62x | 4.44x / 3.23x | 5.34x / 3.41x |

## 詳細

| Group | Source | L | n_pos | recall@1 | recall@4 | recall@16 | mean max@16 | mean top-1 | zero-hit@16 | 候補あり率 | 平均候補数 | speedup oracle | speedup top-1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toolcall/test/known | ngram | 2 | 617 | 29.0% | 44.9% | 53.8% | 1.30 | 0.80 | 21.9% | 94.2% | 5.7 | 1.63x | 1.33x |
| toolcall/test/known | ngram | 4 | 617 | 10.4% | 20.3% | 21.5% | 1.81 | 1.07 | 22.0% | 94.2% | 6.5 | 1.94x | 1.52x |
| toolcall/test/known | ngram | 8 | 617 | 0.6% | 1.3% | 1.3% | 2.01 | 1.13 | 22.4% | 94.2% | 7.8 | 1.94x | 1.55x |
| toolcall/test/known | ngram | 16 | 617 | 0.0% | 0.0% | 0.0% | 2.02 | 1.13 | 22.2% | 94.2% | 8.4 | 1.90x | 1.50x |
| toolcall/test/known | grammar | 2 | 617 | 16.8% | 23.5% | 23.5% | 0.64 | 0.54 | 59.2% | 49.9% | 1.5 | 1.38x | 1.25x |
| toolcall/test/known | grammar | 4 | 617 | 3.6% | 10.8% | 10.8% | 0.89 | 0.67 | 59.2% | 49.9% | 1.7 | 1.45x | 1.31x |
| toolcall/test/known | grammar | 8 | 617 | 0.0% | 0.0% | 0.0% | 0.93 | 0.64 | 59.2% | 49.9% | 1.8 | 1.54x | 1.25x |
| toolcall/test/known | grammar | 16 | 617 | 0.0% | 0.0% | 0.0% | 0.93 | 0.64 | 59.2% | 49.9% | 1.8 | 1.52x | 1.23x |
| toolcall/test/known | corpus | 2 | 617 | 67.8% | 75.7% | 79.1% | 1.63 | 1.43 | 13.9% | 92.1% | 21.1 | 1.92x | 1.75x |
| toolcall/test/known | corpus | 4 | 617 | 51.3% | 61.8% | 68.2% | 2.92 | 2.40 | 14.1% | 92.1% | 30.4 | 2.62x | 2.18x |
| toolcall/test/known | corpus | 8 | 617 | 33.3% | 44.9% | 56.6% | 4.88 | 3.65 | 13.9% | 92.1% | 40.6 | 3.23x | 2.67x |
| toolcall/test/known | corpus | 16 | 617 | 18.6% | 29.0% | 39.7% | 7.15 | 4.83 | 13.9% | 92.1% | 55.1 | 3.77x | 2.91x |
| toolcall/test/known | schema | 2 | 617 | 51.9% | 68.2% | 70.2% | 1.47 | 1.17 | 20.4% | 86.5% | 3.1 | 1.87x | 1.66x |
| toolcall/test/known | schema | 4 | 617 | 39.7% | 55.3% | 57.5% | 2.59 | 1.95 | 20.9% | 86.5% | 4.9 | 2.43x | 1.97x |
| toolcall/test/known | schema | 8 | 617 | 24.5% | 36.5% | 36.9% | 4.01 | 2.93 | 20.7% | 86.5% | 6.8 | 2.80x | 2.27x |
| toolcall/test/known | schema | 16 | 617 | 1.6% | 4.1% | 4.1% | 5.07 | 3.51 | 20.7% | 86.5% | 7.9 | 3.11x | 2.43x |
| toolcall/test/known | finite | 2 | 617 | 70.9% | 84.8% | 90.3% | 1.81 | 1.50 | 6.0% | 99.7% | 31.4 | 2.07x | 1.82x |
| toolcall/test/known | finite | 4 | 617 | 54.0% | 69.5% | 82.0% | 3.35 | 2.56 | 6.5% | 99.7% | 43.5 | 3.01x | 2.30x |
| toolcall/test/known | finite | 8 | 617 | 32.5% | 48.6% | 70.0% | 5.75 | 3.87 | 6.3% | 99.7% | 57.1 | 4.03x | 3.03x |
| toolcall/test/known | finite | 16 | 617 | 18.6% | 29.0% | 40.7% | 7.78 | 5.00 | 6.2% | 99.7% | 73.2 | 4.89x | 3.26x |
| toolcall/test/known | dflash | 2 | 617 | 86.9% | 86.9% | 86.9% | 1.78 | 1.76 | 6.5% | 100.0% | 4.0 | 1.97x | 1.96x |
| toolcall/test/known | dflash | 4 | 617 | 77.6% | 77.6% | 77.6% | 3.26 | 3.24 | 6.5% | 100.0% | 4.0 | 2.65x | 2.64x |
| toolcall/test/known | dflash | 8 | 617 | 44.4% | 44.4% | 44.4% | 5.14 | 5.13 | 6.5% | 100.0% | 4.0 | 3.34x | 3.29x |
| toolcall/test/known | dflash | 16 | 617 | 0.0% | 0.0% | 0.0% | 6.07 | 6.06 | 6.5% | 100.0% | 4.0 | 3.39x | 3.36x |
| toolcall/test/known | union | 2 | 617 | 86.9% | 86.9% | 95.5% | 1.89 | 1.76 | 3.2% | 100.0% | 12.1 | 2.13x | 1.96x |
| toolcall/test/known | union | 4 | 617 | 77.6% | 77.6% | 92.8% | 3.61 | 3.24 | 3.2% | 100.0% | 13.3 | 3.25x | 2.64x |
| toolcall/test/known | union | 8 | 617 | 44.4% | 44.4% | 78.8% | 6.41 | 5.13 | 3.2% | 100.0% | 13.9 | 4.51x | 3.29x |
| toolcall/test/known | union | 16 | 617 | 0.0% | 0.0% | 34.1% | 8.97 | 6.06 | 3.2% | 100.0% | 14.4 | 5.58x | 3.36x |
| toolcall/test/unknown | ngram | 2 | 677 | 25.1% | 39.7% | 48.6% | 1.18 | 0.71 | 28.8% | 90.0% | 5.6 | 1.54x | 1.28x |
| toolcall/test/unknown | ngram | 4 | 677 | 10.4% | 18.5% | 18.8% | 1.60 | 0.97 | 29.5% | 90.0% | 6.7 | 1.74x | 1.40x |
| toolcall/test/unknown | ngram | 8 | 677 | 0.0% | 1.5% | 1.5% | 1.82 | 1.03 | 29.7% | 90.0% | 8.1 | 1.79x | 1.44x |
| toolcall/test/unknown | ngram | 16 | 677 | 0.0% | 0.0% | 0.0% | 1.82 | 1.03 | 30.1% | 90.0% | 9.1 | 1.74x | 1.40x |
| toolcall/test/unknown | grammar | 2 | 677 | 13.9% | 19.5% | 19.5% | 0.54 | 0.45 | 64.8% | 44.3% | 1.3 | 1.30x | 1.20x |
| toolcall/test/unknown | grammar | 4 | 677 | 2.9% | 8.8% | 8.8% | 0.75 | 0.56 | 64.8% | 44.3% | 1.4 | 1.36x | 1.25x |
| toolcall/test/unknown | grammar | 8 | 677 | 0.0% | 0.0% | 0.0% | 0.78 | 0.54 | 64.8% | 44.3% | 1.6 | 1.43x | 1.21x |
| toolcall/test/unknown | grammar | 16 | 677 | 0.0% | 0.0% | 0.0% | 0.78 | 0.54 | 64.8% | 44.3% | 1.6 | 1.41x | 1.19x |
| toolcall/test/unknown | corpus | 2 | 677 | 35.9% | 42.2% | 45.1% | 1.04 | 0.84 | 39.4% | 78.4% | 18.8 | 1.50x | 1.33x |
| toolcall/test/unknown | corpus | 4 | 677 | 19.1% | 23.2% | 27.2% | 1.62 | 1.21 | 39.6% | 78.4% | 27.2 | 1.69x | 1.46x |
| toolcall/test/unknown | corpus | 8 | 677 | 5.6% | 8.4% | 13.6% | 2.14 | 1.43 | 39.7% | 78.4% | 36.0 | 1.84x | 1.52x |
| toolcall/test/unknown | corpus | 16 | 677 | 3.7% | 3.7% | 7.2% | 2.63 | 1.69 | 40.5% | 78.4% | 48.6 | 1.81x | 1.50x |
| toolcall/test/unknown | schema | 2 | 677 | 48.1% | 63.5% | 64.7% | 1.36 | 1.09 | 26.4% | 79.3% | 3.1 | 1.74x | 1.57x |
| toolcall/test/unknown | schema | 4 | 677 | 33.5% | 48.1% | 51.1% | 2.37 | 1.75 | 26.7% | 79.3% | 5.3 | 2.12x | 1.83x |
| toolcall/test/unknown | schema | 8 | 677 | 18.6% | 31.3% | 31.3% | 3.61 | 2.58 | 26.9% | 79.3% | 7.8 | 2.43x | 2.06x |
| toolcall/test/unknown | schema | 16 | 677 | 3.7% | 5.8% | 6.1% | 4.60 | 3.00 | 26.9% | 79.3% | 9.0 | 2.59x | 2.15x |
| toolcall/test/unknown | finite | 2 | 677 | 52.4% | 62.9% | 74.1% | 1.55 | 1.20 | 16.8% | 96.9% | 28.9 | 1.84x | 1.57x |
| toolcall/test/unknown | finite | 4 | 677 | 33.5% | 39.2% | 58.2% | 2.68 | 1.88 | 16.8% | 96.9% | 40.5 | 2.35x | 1.84x |
| toolcall/test/unknown | finite | 8 | 677 | 12.5% | 16.4% | 38.5% | 4.14 | 2.49 | 17.1% | 96.9% | 53.5 | 2.78x | 2.10x |
| toolcall/test/unknown | finite | 16 | 677 | 3.7% | 3.7% | 11.4% | 4.31 | 2.81 | 19.9% | 96.9% | 68.3 | 2.93x | 2.15x |
| toolcall/test/unknown | dflash | 2 | 677 | 82.6% | 84.0% | 84.0% | 1.76 | 1.71 | 5.9% | 100.0% | 4.0 | 2.01x | 1.98x |
| toolcall/test/unknown | dflash | 4 | 677 | 71.3% | 72.1% | 72.1% | 3.15 | 3.09 | 5.9% | 100.0% | 4.0 | 2.73x | 2.69x |
| toolcall/test/unknown | dflash | 8 | 677 | 41.5% | 41.9% | 41.9% | 4.92 | 4.84 | 5.9% | 100.0% | 4.0 | 3.40x | 3.29x |
| toolcall/test/unknown | dflash | 16 | 677 | 0.0% | 0.0% | 0.0% | 5.91 | 5.82 | 5.9% | 100.0% | 4.0 | 3.46x | 3.35x |
| toolcall/test/unknown | union | 2 | 677 | 82.6% | 84.0% | 91.2% | 1.85 | 1.71 | 3.2% | 100.0% | 11.5 | 2.10x | 1.98x |
| toolcall/test/unknown | union | 4 | 677 | 71.3% | 72.1% | 81.2% | 3.40 | 3.09 | 3.2% | 100.0% | 12.5 | 2.99x | 2.69x |
| toolcall/test/unknown | union | 8 | 677 | 41.5% | 41.9% | 55.9% | 5.59 | 4.84 | 3.2% | 100.0% | 13.0 | 3.94x | 3.29x |
| toolcall/test/unknown | union | 16 | 677 | 0.0% | 0.0% | 5.8% | 7.03 | 5.82 | 3.2% | 100.0% | 13.5 | 4.40x | 3.35x |
| toolcall/train | ngram | 2 | 2449 | 28.6% | 45.3% | 54.1% | 1.28 | 0.79 | 24.5% | 91.4% | 5.4 | 1.62x | 1.33x |
| toolcall/train | ngram | 4 | 2449 | 11.8% | 20.5% | 21.4% | 1.78 | 1.08 | 24.9% | 91.4% | 6.2 | 1.87x | 1.49x |
| toolcall/train | ngram | 8 | 2449 | 1.3% | 2.7% | 2.8% | 2.02 | 1.16 | 25.0% | 91.4% | 7.2 | 1.89x | 1.52x |
| toolcall/train | ngram | 16 | 2449 | 0.0% | 0.0% | 0.0% | 2.06 | 1.17 | 24.9% | 91.4% | 7.7 | 1.85x | 1.48x |
| toolcall/train | grammar | 2 | 2449 | 16.2% | 22.7% | 22.7% | 0.62 | 0.53 | 60.1% | 49.8% | 1.5 | 1.37x | 1.25x |
| toolcall/train | grammar | 4 | 2449 | 3.5% | 10.4% | 10.4% | 0.87 | 0.66 | 60.1% | 49.8% | 1.6 | 1.44x | 1.31x |
| toolcall/train | grammar | 8 | 2449 | 0.0% | 0.0% | 0.0% | 0.90 | 0.62 | 60.1% | 49.8% | 1.8 | 1.52x | 1.24x |
| toolcall/train | grammar | 16 | 2449 | 0.0% | 0.0% | 0.0% | 0.90 | 0.62 | 60.1% | 49.8% | 1.8 | 1.50x | 1.22x |
| toolcall/train | corpus | 2 | 2449 | 64.1% | 70.7% | 75.0% | 1.55 | 1.37 | 17.4% | 89.4% | 18.9 | 1.85x | 1.71x |
| toolcall/train | corpus | 4 | 2449 | 49.0% | 57.4% | 63.7% | 2.77 | 2.31 | 17.6% | 89.4% | 28.0 | 2.40x | 2.03x |
| toolcall/train | corpus | 8 | 2449 | 29.6% | 39.8% | 49.5% | 4.51 | 3.41 | 17.8% | 89.4% | 35.5 | 2.98x | 2.47x |
| toolcall/train | corpus | 16 | 2449 | 15.3% | 27.5% | 33.4% | 6.45 | 4.46 | 18.0% | 89.4% | 43.0 | 3.32x | 2.72x |
| toolcall/train | schema | 2 | 2449 | 53.7% | 68.3% | 70.2% | 1.46 | 1.20 | 21.5% | 84.5% | 3.1 | 1.85x | 1.68x |
| toolcall/train | schema | 4 | 2449 | 41.4% | 55.5% | 58.1% | 2.59 | 2.01 | 21.5% | 84.5% | 4.8 | 2.41x | 2.01x |
| toolcall/train | schema | 8 | 2449 | 28.7% | 34.8% | 34.9% | 3.95 | 3.10 | 21.5% | 84.5% | 6.9 | 2.74x | 2.32x |
| toolcall/train | schema | 16 | 2449 | 2.3% | 3.7% | 4.0% | 4.97 | 3.85 | 21.5% | 84.5% | 7.7 | 3.01x | 2.51x |
| toolcall/train | finite | 2 | 2449 | 69.6% | 80.7% | 86.6% | 1.75 | 1.49 | 8.7% | 98.5% | 28.9 | 2.01x | 1.83x |
| toolcall/train | finite | 4 | 2449 | 53.1% | 64.8% | 77.8% | 3.21 | 2.49 | 9.0% | 98.5% | 40.6 | 2.82x | 2.26x |
| toolcall/train | finite | 8 | 2449 | 31.0% | 45.1% | 64.7% | 5.47 | 3.63 | 8.9% | 98.5% | 51.3 | 3.71x | 2.74x |
| toolcall/train | finite | 16 | 2449 | 15.3% | 27.5% | 35.1% | 7.23 | 4.63 | 9.1% | 98.5% | 60.2 | 4.22x | 2.98x |
| toolcall/train | dflash | 2 | 2449 | 84.9% | 85.4% | 85.4% | 1.76 | 1.73 | 6.7% | 100.0% | 4.0 | 1.98x | 1.94x |
| toolcall/train | dflash | 4 | 2449 | 76.6% | 76.9% | 76.9% | 3.21 | 3.18 | 6.7% | 100.0% | 4.0 | 2.66x | 2.62x |
| toolcall/train | dflash | 8 | 2449 | 46.0% | 46.0% | 46.0% | 5.11 | 5.08 | 6.7% | 100.0% | 4.0 | 3.29x | 3.23x |
| toolcall/train | dflash | 16 | 2449 | 0.0% | 0.0% | 0.0% | 6.16 | 6.12 | 6.7% | 100.0% | 4.0 | 3.45x | 3.41x |
| toolcall/train | union | 2 | 2449 | 84.9% | 85.4% | 94.2% | 1.88 | 1.73 | 3.5% | 100.0% | 11.8 | 2.12x | 1.94x |
| toolcall/train | union | 4 | 2449 | 76.6% | 76.9% | 89.9% | 3.54 | 3.18 | 3.5% | 100.0% | 13.0 | 3.13x | 2.62x |
| toolcall/train | union | 8 | 2449 | 46.0% | 46.0% | 79.0% | 6.29 | 5.08 | 3.5% | 100.0% | 13.5 | 4.44x | 3.23x |
| toolcall/train | union | 16 | 2449 | 0.0% | 0.0% | 33.3% | 8.66 | 6.12 | 3.5% | 100.0% | 14.1 | 5.34x | 3.41x |
