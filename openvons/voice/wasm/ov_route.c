/* openvons の振り分け (候補の絞り込み → CTC 採点 → 該当なし付き校正 → 判断) を C で。WebAssembly (ブラウザ) と
 * ネイティブ (Python との一致試験) の両方でビルドする。音声側は komimi (Conformer-CTC) の C エンジンをそのまま使う。
 *
 * Python の openvons.voice.engine.Recognizer (+ grammar.CommandSet.shortlist / none_calibration / core.decision) と
 * 同じ手順・同じ定数で、rapidfuzz の Levenshtein.normalized_similarity と fuzz.partial_ratio(_alignment) も
 * 参照実装 (rapidfuzz/fuzz_py.py) の窓の順番と同点の扱いまで合わせてある。
 *
 *   ovk_new(kmm, n)                      モデル (.kmm) を読む
 *   ovk_set_vocab_scores(h, scores, n)   語彙の unigram スコア (komimi の models/ja1024_vocab.json) — 候補カナ列の分割用
 *   ovk_set_commands(h, blob, n)         コマンド集合 (仮説ごとに 意味 id・危険度・フラグ・カナ)
 *   ovk_set_params(h, p, n)              校正・閾値・絞り込みの定数 (並びは下の P_*)
 *   ovk_route(h, pcm, n)                 16 kHz float PCM → 判断の JSON
 *   ovk_route_logprobs(h, lp, T)         CTC の log-softmax (T, vocab) から判断の JSON (一致試験用)
 */
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "km.h"
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define API EMSCRIPTEN_KEEPALIVE
static double now_ms(void) { return emscripten_get_now(); }
#else
#define API __attribute__((visibility("default")))
#include <time.h>
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
#endif

#define NEG (-1e30)

/* ---------------------------------------------------------------- UTF-8 */
static int utf8_decode(const uint8_t *s, int n, uint32_t *out) {
    int k = 0;
    for (int i = 0; i < n;) {
        uint32_t c = s[i];
        if (c < 0x80) { i += 1; }
        else if ((c >> 5) == 6 && i + 1 < n) { c = ((c & 31) << 6) | (s[i + 1] & 63); i += 2; }
        else if ((c >> 4) == 14 && i + 2 < n) { c = ((c & 15) << 12) | ((s[i + 1] & 63) << 6) | (s[i + 2] & 63); i += 3; }
        else if (i + 3 < n) { c = ((c & 7) << 18) | ((s[i + 1] & 63) << 12) | ((s[i + 2] & 63) << 6) | (s[i + 3] & 63); i += 4; }
        else break;
        out[k++] = c;
    }
    return k;
}
static int utf8_put(char *o, uint32_t c) {
    if (c < 0x80) { o[0] = (char)c; return 1; }
    if (c < 0x800) { o[0] = (char)(0xC0 | (c >> 6)); o[1] = (char)(0x80 | (c & 63)); return 2; }
    if (c < 0x10000) { o[0] = (char)(0xE0 | (c >> 12)); o[1] = (char)(0x80 | ((c >> 6) & 63)); o[2] = (char)(0x80 | (c & 63)); return 3; }
    o[0] = (char)(0xF0 | (c >> 18)); o[1] = (char)(0x80 | ((c >> 12) & 63)); o[2] = (char)(0x80 | ((c >> 6) & 63)); o[3] = (char)(0x80 | (c & 63)); return 4;
}

/* ---------------------------------------------------------------- 可変長バッファ (JSON) */
typedef struct { char *p; size_t n, cap; } buf_t;
static void bput(buf_t *b, const char *fmt, ...) {
    for (;;) {
        va_list ap; va_start(ap, fmt);
        int w = vsnprintf(b->p + b->n, b->cap - b->n, fmt, ap); va_end(ap);
        if (w >= 0 && (size_t)w < b->cap - b->n) { b->n += (size_t)w; return; }
        b->cap = b->cap * 2 + (size_t)(w > 0 ? w : 256) + 256; b->p = (char *)realloc(b->p, b->cap);
    }
}
static void bstr(buf_t *b, const uint32_t *s, int n) {      /* JSON 文字列 (カナなのでエスケープは " と \ だけ) */
    char tmp[8]; bput(b, "\"");
    for (int i = 0; i < n; i++) { if (s[i] == '"' || s[i] == '\\') bput(b, "\\"); int w = utf8_put(tmp, s[i]); tmp[w] = 0; bput(b, "%s", tmp); }
    bput(b, "\"");
}

/* ---------------------------------------------------------------- カナ (openvons.voice.kana と同じ表) */
static int in_set(uint32_t c, const char *set) {   /* set は UTF-8 のカナ列 */
    uint32_t t[64]; int n = utf8_decode((const uint8_t *)set, (int)strlen(set), t);
    for (int i = 0; i < n; i++) if (t[i] == c) return 1;
    return 0;
}
static const char *O_ROW = "オコソトノホモヨロヲゴゾドボポォョ", *U_ROW = "ウクスツヌフムユルグズヅブプゥュ", *E_ROW = "エケセテネヘメレゲゼデベペェ", *SMALL = "ァィゥェォャュョヮ";
#define CH_U 0x30A6
#define CH_I 0x30A4
#define CH_BAR 0x30FC
#define CH_SOKUON 0x30C3
static uint32_t replace_ch(uint32_t c) {
    switch (c) {
        case 0x30F2: return 0x30AA; /* ヲ→オ */ case 0x30C2: return 0x30B8; /* ヂ→ジ */ case 0x30C5: return 0x30BA; /* ヅ→ズ */
        case 0x30F0: return 0x30A4; /* ヰ→イ */ case 0x30F1: return 0x30A8; /* ヱ→エ */ case 0x30F5: return 0x30AB; /* ヵ→カ */
        case 0x30F6: return 0x30B1; /* ヶ→ケ */ case 0x309D: case 0x309E: return 0; /* ゝゞ → 消す */
        default: return c;
    }
}
/* normalize(): 置換 → 長音化 (オ段/ウ段+ウ、エ段+イ → ー、ーー → ー) → 先頭の ー・ッ を落とす。komimi の出力はカタカナだけなので句読点の除去は空白だけ */
static int kana_normalize(const uint32_t *s, int n, uint32_t *out) {
    int k = 0;
    for (int i = 0; i < n; i++) {
        uint32_t c = replace_ch(s[i]);
        if (!c || c == ' ' || c == 0x3000 || c == 0x2581) continue;
        if (k) {
            uint32_t p = out[k - 1];
            if (c == CH_U && (in_set(p, O_ROW) || in_set(p, U_ROW))) { out[k++] = CH_BAR; continue; }
            if (c == CH_I && in_set(p, E_ROW)) { out[k++] = CH_BAR; continue; }
            if (c == CH_BAR && p == CH_BAR) continue;
        }
        out[k++] = c;
    }
    int st = 0; while (st < k && (out[st] == CH_BAR || out[st] == CH_SOKUON)) st++;
    if (st) { memmove(out, out + st, sizeof(uint32_t) * (size_t)(k - st)); k -= st; }
    return k;
}
static int n_morae(const uint32_t *s, int n) { int m = 0; for (int i = 0; i < n; i++) if (!(in_set(s[i], SMALL) && m)) m++; return m; }
static int mora_split(const uint32_t *s, int n, int *start) {   /* start[j] = j 番目のモーラの先頭位置、戻り値はモーラ数 */
    int m = 0; for (int i = 0; i < n; i++) if (!(in_set(s[i], SMALL) && m)) start[m++] = i;
    start[m] = n; return m;
}
static int is_degenerate(const uint32_t *s, int n) {             /* engine._is_degenerate (max_repeat 6) */
    int *st = (int *)malloc(sizeof(int) * (size_t)(n + 2)); int m = mora_split(s, n, st), res = 0;
    #define MOR_EQ(a, b) ((st[(a) + 1] - st[a]) == (st[(b) + 1] - st[b]) && !memcmp(s + st[a], s + st[b], sizeof(uint32_t) * (size_t)(st[(a) + 1] - st[a])))
    for (int w = 1; w <= 2 && !res; w++) {
        int run = 1;
        for (int i = w; i < m; i++) {
            int eq = (i + w <= m);
            for (int j = 0; j < w && eq; j++) eq = MOR_EQ(i - w + j, i + j);
            if (eq) { if (++run >= 6) { res = 1; break; } } else run = 1;
        }
    }
    #undef MOR_EQ
    free(st); return res;
}

/* ---------------------------------------------------------------- 文字列の類似度 (rapidfuzz 互換) */
static int lev_dist(const uint32_t *a, int la, const uint32_t *b, int lb, int *row) {
    for (int j = 0; j <= lb; j++) row[j] = j;
    for (int i = 1; i <= la; i++) {
        int prev = row[0]; row[0] = i;
        for (int j = 1; j <= lb; j++) {
            int cur = row[j], v = prev + (a[i - 1] != b[j - 1]);
            if (row[j] + 1 < v) v = row[j] + 1;
            if (row[j - 1] + 1 < v) v = row[j - 1] + 1;
            row[j] = v; prev = cur;
        }
    }
    return row[lb];
}
static double lev_norm_sim(const uint32_t *a, int la, const uint32_t *b, int lb, int *row) {
    int mx = la > lb ? la : lb; if (!mx) return 1.0;
    return 1.0 - (double)lev_dist(a, la, b, lb, row) / mx;
}
static int lcs(const uint32_t *a, int la, const uint32_t *b, int lb, int *row) {
    memset(row, 0, sizeof(int) * (size_t)(lb + 1));
    for (int i = 1; i <= la; i++) {
        int prev = 0;
        for (int j = 1; j <= lb; j++) {
            int cur = row[j];
            row[j] = (a[i - 1] == b[j - 1]) ? prev + 1 : (row[j] > row[j - 1] ? row[j] : row[j - 1]);
            prev = cur;
        }
    }
    return row[lb];
}
static double indel_norm_sim(const uint32_t *a, int la, const uint32_t *b, int lb, double cutoff, int *row) {
    if (la + lb == 0) return 1.0;
    double s = 2.0 * lcs(a, la, b, lb, row) / (la + lb);
    return s >= cutoff ? s : 0.0;
}
typedef struct { double score; int ss, se, ds, de; } align_t;
static int has_ch(const uint32_t *s, int n, uint32_t c) { for (int i = 0; i < n; i++) if (s[i] == c) return 1; return 0; }
/* rapidfuzz (C++ 版) の partial_ratio_impl (len(s1) <= len(s2)、短い needle)。同じ長さの窓は二分探索の順で調べ、
 * 同点は先に見た窓を採る (Python 版 fuzz_py とは同点の扱いが違う。Recognizer が実際に使うのは C++ 版)。
 * 乱択 20 万組で rapidfuzz 3.14 と score・窓まで一致を確認した。 */
static align_t partial_impl(const uint32_t *s1, int l1, const uint32_t *s2, int l2, double cutoff, int *row) {
    align_t r = {0, 0, l1, 0, l1};
    if (l2 > l1) {
        int maximum = 2 * l1, nw = l2 - l1, best = -1;
        long cutoff_dist = (long)ceil(maximum * (1.0 - cutoff / 100.0)) + 1;
        int *sc = (int *)malloc(sizeof(int) * (size_t)nw); for (int i = 0; i < nw; i++) sc[i] = -1;
        int *win = (int *)malloc(sizeof(int) * (size_t)(4 * nw + 4)), *nwin = (int *)malloc(sizeof(int) * (size_t)(4 * nw + 4)), nwn = 1, nnew;
        win[0] = 0; win[1] = nw - 1;
        while (nwn) {
            nnew = 0;
            for (int w = 0; w < nwn; w++) {
                int a = win[2 * w], b = win[2 * w + 1], pp[2] = {a, b};
                for (int q = 0; q < 2; q++) {
                    int p = pp[q];
                    if (sc[p] < 0) {
                        sc[p] = l1 + l1 - 2 * lcs(s1, l1, s2 + p, l1, row);
                        if (sc[p] < cutoff_dist) {
                            cutoff_dist = best = sc[p]; r.ds = p; r.de = p + l1;
                            if (best == 0) { r.score = 100; free(sc); free(win); free(nwin); return r; }
                        }
                    }
                }
                int d = b - a; if (d == 1) continue;
                int known = sc[a] > sc[b] ? sc[a] - sc[b] : sc[b] - sc[a];
                int imp = (d - known / 2) / 2 * 2;
                long mn = (long)(sc[a] < sc[b] ? sc[a] : sc[b]) - imp;
                if (mn < cutoff_dist) { int c = d / 2; nwin[2 * nnew] = a; nwin[2 * nnew + 1] = a + c; nnew++; nwin[2 * nnew] = a + c; nwin[2 * nnew + 1] = b; nnew++; }
            }
            int *t = win; win = nwin; nwin = t; nwn = nnew;
        }
        if (best >= 0) { double v = (1.0 - (double)best / maximum) * 100; if (v >= cutoff) cutoff = r.score = v; }
        free(sc); free(win); free(nwin);
    }
    for (int i = 1; i < l1; i++) {
        if (!has_ch(s1, l1, s2[i - 1])) continue;
        double v = 100.0 * indel_norm_sim(s1, l1, s2, i, cutoff / 100.0, row);
        if (v > r.score) { r.score = cutoff = v; r.ds = 0; r.de = i; if (v == 100.0) return r; }
    }
    for (int i = l2 - l1; i < l2; i++) {
        if (i < 0 || !has_ch(s1, l1, s2[i])) continue;
        double v = 100.0 * indel_norm_sim(s1, l1, s2 + i, l2 - i, cutoff / 100.0, row);
        if (v > r.score) { r.score = cutoff = v; r.ds = i; r.de = l2; if (v == 100.0) return r; }
    }
    return r;
}
/* rapidfuzz の partial_ratio_alignment。戻り値 score < 0 は None (cutoff 未満) */
static align_t partial_align(const uint32_t *s1, int l1, const uint32_t *s2, int l2, double cutoff, int *row) {
    align_t r;
    if (l1 > l2) {
        r = partial_align(s2, l2, s1, l1, cutoff, row);
        if (r.score >= 0) { align_t t = {r.score, r.ds, r.de, r.ss, r.se}; r = t; }
        return r;
    }
    if (!l1 || !l2) { align_t t = {(l1 == l2) ? 100.0 : 0.0, 0, l1, 0, l1}; r = t; }
    else {
        r = partial_impl(s1, l1, s2, l2, cutoff, row);
        if (r.score != 100 && l1 == l2) {
            double c2 = cutoff > r.score ? cutoff : r.score;
            align_t r2 = partial_impl(s2, l2, s1, l1, c2, row);
            if (r2.score > r.score) { align_t t = {r2.score, r2.ds, r2.de, r2.ss, r2.se}; r = t; }
        }
    }
    if (r.score < cutoff) r.score = -1;
    return r;
}

/* ---------------------------------------------------------------- トークナイザ (unigram Viterbi、komimi/engine.py の Tokenizer と同じ) */
typedef struct { uint64_t key; int id; } hent;
typedef struct {
    int n, maxlen, unk; double *score; hent *tab; int cap;
} tok_t;
static uint64_t hash_cps(const uint32_t *s, int n) { uint64_t h = 1469598103934665603ULL; for (int i = 0; i < n; i++) { h ^= s[i]; h *= 1099511628211ULL; } return h ^ (uint64_t)n * 0x9E3779B97F4A7C15ULL; }
static int tok_find(const tok_t *t, const uint32_t *s, int n) {
    uint64_t h = hash_cps(s, n); int i = (int)(h & (uint64_t)(t->cap - 1));
    while (t->tab[i].id >= 0) { if (t->tab[i].key == h) return t->tab[i].id; i = (i + 1) & (t->cap - 1); }
    return -1;
}
static int tok_encode(const tok_t *t, const uint32_t *text, int n, int *out) {
    int N = n + 1; uint32_t *s = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)N); s[0] = 0x2581; memcpy(s + 1, text, sizeof(uint32_t) * (size_t)n);
    double *best = (double *)malloc(sizeof(double) * (size_t)(N + 1)); int *bj = (int *)malloc(sizeof(int) * (size_t)(N + 1)), *bp = (int *)malloc(sizeof(int) * (size_t)(N + 1));
    best[0] = 0; for (int i = 1; i <= N; i++) best[i] = -INFINITY;
    for (int i = 1; i <= N; i++) {
        for (int L = 1; L <= t->maxlen && L <= i; L++) {
            if (best[i - L] == -INFINITY) continue;
            int pid = tok_find(t, s + i - L, L); if (pid < 0 || pid == t->unk) continue;
            double c = best[i - L] + t->score[pid];
            if (c > best[i] + 1e-12) { best[i] = c; bj[i] = i - L; bp[i] = pid; }
        }
        if (best[i] == -INFINITY) { best[i] = best[i - 1] - 20.0; bj[i] = i - 1; bp[i] = t->unk; }
    }
    int k = 0; for (int i = N; i > 0; i = bj[i]) out[k++] = bp[i];
    for (int a = 0, b = k - 1; a < b; a++, b--) { int x = out[a]; out[a] = out[b]; out[b] = x; }
    free(s); free(best); free(bj); free(bp); return k;
}

/* ---------------------------------------------------------------- CTC */
static double logadd(double a, double b) { if (a <= NEG / 2) return b; if (b <= NEG / 2) return a; double m = a > b ? a : b; return m + log(exp(a - m) + exp(b - m)); }
static double ctc_score(const float *lp, int T, int V, const int *y, int L, double *buf) {
    int blank = V - 1, S = 2 * L + 1; double *a = buf, *b = buf + S;
    #define LAB(s) (((s) & 1) ? y[(s) >> 1] : blank)
    for (int s = 0; s < S; s++) a[s] = NEG;
    a[0] = lp[blank]; if (L > 0) a[1] = lp[y[0]];
    for (int t = 1; t < T; t++) {
        const float *row = lp + (size_t)t * V;
        for (int s = 0; s < S; s++) {
            double v = a[s];
            if (s >= 1) v = logadd(v, a[s - 1]);
            if (s >= 2 && (s & 1) && y[s >> 1] != y[(s >> 1) - 1]) v = logadd(v, a[s - 2]);
            b[s] = v <= NEG / 2 ? NEG : v + row[LAB(s)];
        }
        double *tmp = a; a = b; b = tmp;
    }
    #undef LAB
    double r = a[S - 1]; if (S >= 2) r = logadd(r, a[S - 2]);
    return r;
}

/* ---------------------------------------------------------------- 本体 */
enum { P_T, P_B0, P_B1, P_G, P_EXEC, P_CONF, P_EXEC_MED, P_CLEAR_MIN, P_CLEAR_RATIO, P_CLEAR_NONE, P_ANS_YES, P_ANS_NO,
       P_K, P_EMBED, P_EMB_RATIO, P_EMB_RESID, P_EMB_MORAE, P_EMB_PEN, P_N };
static const double P_DEFAULT[P_N] = {2.5, 4.0, 1.4, 1.0, 0.85, 0.40, 0.95, 0.65, 3.0, 0.20, 0.60, 0.40, 16, 1, 0.3, 14, 4, 0.5};

typedef struct { uint32_t *k; int n; int meaning; int risk; int flags; } hyp_t;     /* flags: 1 allow_embed, 2 confirmable, 4 positive */
typedef struct {
    km_model m; uint8_t *mbuf; tok_t tok;
    hyp_t *h; int nh, nmean; uint32_t *pool; int *bare; int nbare;
    double P[P_N];
    buf_t out; float *lp; int lp_cap;
} ovk_t;

API void *ovk_new(const uint8_t *kmm, int n) {
    ovk_t *o = (ovk_t *)calloc(1, sizeof *o);
    o->mbuf = (uint8_t *)malloc((size_t)n + 64); memcpy(o->mbuf, kmm, (size_t)n);
    if (km_load(&o->m, o->mbuf, (size_t)n) != 0) { free(o->mbuf); free(o); return NULL; }
    memcpy(o->P, P_DEFAULT, sizeof o->P);
    /* 語彙表 (piece → id)。スコアは ovk_set_vocab_scores で入るまで 0 (= 最少 piece 数の分割) */
    tok_t *t = &o->tok; t->n = o->m.n_tokens; t->unk = 0; t->score = (double *)calloc((size_t)t->n, sizeof(double));
    t->cap = 1; while (t->cap < t->n * 4) t->cap <<= 1;
    t->tab = (hent *)malloc(sizeof(hent) * (size_t)t->cap); for (int i = 0; i < t->cap; i++) t->tab[i].id = -1;
    uint32_t cp[64];
    for (int i = 0; i < t->n; i++) {
        int L = utf8_decode((const uint8_t *)o->m.tok[i], o->m.tok_len[i], cp); if (L <= 0 || L > 60) continue;
        if (L > t->maxlen) t->maxlen = L;
        uint64_t hsh = hash_cps(cp, L); int j = (int)(hsh & (uint64_t)(t->cap - 1));
        while (t->tab[j].id >= 0) j = (j + 1) & (t->cap - 1);
        t->tab[j].key = hsh; t->tab[j].id = i;
    }
    o->out.cap = 4096; o->out.p = (char *)malloc(o->out.cap);
    return o;
}
API int ovk_vocab(void *p) { return ((ovk_t *)p)->m.vocab; }
API int ovk_set_vocab_scores(void *p, const float *s, int n) {
    ovk_t *o = (ovk_t *)p; if (n > o->tok.n) n = o->tok.n;
    for (int i = 0; i < n; i++) o->tok.score[i] = s[i];
    return n;
}
API void ovk_set_params(void *p, const double *v, int n) { ovk_t *o = (ovk_t *)p; for (int i = 0; i < n && i < P_N; i++) o->P[i] = v[i]; }

/* blob: u32 n; 仮説ごとに u32 meaning, u8 risk, u8 flags, u16 bytes, UTF-8 カナ */
API int ovk_set_commands(void *p, const uint8_t *blob, int nbytes) {
    ovk_t *o = (ovk_t *)p;
    free(o->h); free(o->pool); free(o->bare); o->h = NULL; o->pool = NULL; o->bare = NULL; o->nh = o->nmean = o->nbare = 0;
    if (nbytes < 4) return -1;
    uint32_t n; memcpy(&n, blob, 4); size_t off = 4;
    o->h = (hyp_t *)calloc(n ? n : 1, sizeof(hyp_t)); o->pool = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)nbytes + 16);
    size_t pk = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (off + 8 > (size_t)nbytes) return -2;
        uint32_t mean; uint16_t nb; memcpy(&mean, blob + off, 4); o->h[i].risk = blob[off + 4]; o->h[i].flags = blob[off + 5]; memcpy(&nb, blob + off + 6, 2); off += 8;
        if (off + nb > (size_t)nbytes) return -3;
        o->h[i].k = o->pool + pk; o->h[i].n = utf8_decode(blob + off, nb, o->h[i].k); pk += (size_t)o->h[i].n; off += nb;
        o->h[i].meaning = (int)mean; if ((int)mean + 1 > o->nmean) o->nmean = (int)mean + 1;
    }
    o->nh = (int)n;
    /* 意味ごとに短い方から 3 表層形 (同じ長さは元の順) = CommandSet.bare_index。意味は初出順に並べる (Python の dict と同じ) */
    int *first = (int *)malloc(sizeof(int) * (size_t)(o->nmean + 1)), *order = (int *)malloc(sizeof(int) * (size_t)(o->nmean + 1)), no = 0;
    for (int m = 0; m < o->nmean; m++) first[m] = -1;
    for (int i = 0; i < o->nh; i++) if (first[o->h[i].meaning] < 0) { first[o->h[i].meaning] = i; order[no++] = o->h[i].meaning; }
    int *cnt = (int *)calloc((size_t)o->nmean + 1, sizeof(int)), *beg = (int *)calloc((size_t)o->nmean + 2, sizeof(int)), *ix = (int *)malloc(sizeof(int) * (size_t)(o->nh + 1));
    for (int i = 0; i < o->nh; i++) cnt[o->h[i].meaning]++;
    for (int m = 0; m < o->nmean; m++) beg[m + 1] = beg[m] + cnt[m];
    memset(cnt, 0, sizeof(int) * (size_t)o->nmean);
    for (int i = 0; i < o->nh; i++) { int m = o->h[i].meaning; ix[beg[m] + cnt[m]++] = i; }
    o->bare = (int *)malloc(sizeof(int) * (size_t)(o->nh + 1));
    for (int q = 0; q < no; q++) {
        int m = order[q], *a = ix + beg[m], c = cnt[m];
        for (int x = 1; x < c; x++) { int v = a[x], y = x - 1; while (y >= 0 && o->h[a[y]].n > o->h[v].n) { a[y + 1] = a[y]; y--; } a[y + 1] = v; }   /* 安定な挿入ソート */
        for (int x = 0; x < c && x < 3; x++) o->bare[o->nbare++] = a[x];
    }
    free(first); free(order); free(cnt); free(beg); free(ix);
    return o->nh;
}

typedef struct { double s; int idx; int seq; } rank_t;
static int rank_cmp(const void *a, const void *b) {   /* score 降順、同点は入れた順 (安定) */
    const rank_t *x = (const rank_t *)a, *y = (const rank_t *)b;
    if (x->s > y->s) return -1; if (x->s < y->s) return 1; return x->seq - y->seq;
}
/* process.extract(limit) と同じ: score 降順・同点は index 昇順で上位 limit */
static int topk(rank_t *all, int n, int limit) { qsort(all, (size_t)n, sizeof(rank_t), rank_cmp); return n < limit ? n : limit; }

static double cal_logit(const double *P, double s, double n, double nf) {
    double cs = s + P[P_B1] * (n < nf ? n : nf) - P[P_G] * (nf - n > 0 ? nf - n : 0);
    return cs / (P[P_T] > 1e-3 ? P[P_T] : 1e-3);
}

static const char *route_lp(ovk_t *o, const float *lp, int T, double t_enc) {
    double t0 = now_ms(); int V = o->m.vocab, blank = V - 1; const double *P = o->P;
    buf_t *b = &o->out; b->n = 0;
    /* 1. 自由認識 (greedy) と、その CTC 対数尤度 */
    int *ids = (int *)malloc(sizeof(int) * (size_t)(T + 1)), nid = 0, prev = blank;
    for (int t = 0; t < T; t++) {
        const float *r = lp + (size_t)t * V; int bi = 0; for (int v = 1; v < V; v++) if (r[v] > r[bi]) bi = v;
        if (bi != blank && bi != prev) ids[nid++] = bi; prev = bi;
    }
    int cap = (o->tok.maxlen + 1) * (T + 64); double *dpb = (double *)malloc(sizeof(double) * (size_t)(4 * (T + 64) + 16));
    double free_lp;
    if (nid) free_lp = ctc_score(lp, T, V, ids, nid, dpb); else { free_lp = 0; for (int t = 0; t < T; t++) free_lp += lp[(size_t)t * V + blank]; }
    uint32_t *raw = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)cap), *fk = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)cap); int nraw = 0;
    for (int i = 0; i < nid; i++) { int id = ids[i]; if (id == o->tok.unk) continue; nraw += utf8_decode((const uint8_t *)o->m.tok[id], o->m.tok_len[id], raw + nraw); }
    int nfk = kana_normalize(raw, nraw, fk);
    double n_free = nid; if (!nid) { int tmp[8]; n_free = tok_encode(&o->tok, fk, nfk, tmp); }
    double t_tr = now_ms();
    bput(b, "{\"free_kana\":"); bstr(b, fk, nfk);
    bput(b, ",\"free_score\":%.4f,\"n_free\":%d,\"frames\":%d", free_lp, (int)n_free, T);
    int degenerate = nfk && is_degenerate(fk, nfk);
    if (!o->nh || !nfk || degenerate) {
        bput(b, ",\"cands\":[],\"meanings\":[],\"none_prob\":1,\"action\":\"none\",\"reason\":\"%s\"", degenerate ? "反復ハルシネーション" : "empty");
        bput(b, ",\"timings_ms\":{\"encode\":%.1f,\"transcribe\":%.1f,\"total\":%.1f}}", t_enc, t_tr - t0, t_enc + now_ms() - t0);
        free(ids); free(dpb); free(raw); free(fk); return b->p;
    }
    /* 2. 絞り込み (CommandSet.shortlist) */
    int K = (int)P[P_K], *sl = (int *)malloc(sizeof(int) * (size_t)(o->nh + 1)), nsl = 0;
    int maxlen = nfk; for (int i = 0; i < o->nh; i++) if (o->h[i].n > maxlen) maxlen = o->h[i].n;
    int *row = (int *)malloc(sizeof(int) * (size_t)(maxlen + 2));
    if (o->nh <= K) { for (int i = 0; i < o->nh; i++) sl[nsl++] = i; }
    else {
        rank_t *full = (rank_t *)malloc(sizeof(rank_t) * (size_t)o->nh);
        for (int i = 0; i < o->nh; i++) { full[i].s = lev_norm_sim(fk, nfk, o->h[i].k, o->h[i].n, row); full[i].idx = i; full[i].seq = i; }
        int nf = topk(full, o->nh, 3 * K);
        rank_t *part = (rank_t *)malloc(sizeof(rank_t) * (size_t)(o->nbare + 1)); int np = 0;
        for (int j = 0; j < o->nbare; j++) {
            const hyp_t *h = &o->h[o->bare[j]];
            align_t al = partial_align(fk, nfk, h->k, h->n, 80.0, row);
            if (al.score >= 0) { part[np].s = al.score; part[np].idx = j; part[np].seq = j; np++; }
        }
        np = topk(part, np, 2 * K);
        int min_len = (int)(0.3 * nfk); if (min_len < 3) min_len = 3;
        rank_t *rk = (rank_t *)malloc(sizeof(rank_t) * (size_t)(nf + np + 1)); int nr = 0;
        for (int i = 0; i < nf; i++) { rk[nr] = full[i]; rk[nr].seq = nr; nr++; }
        for (int i = 0; i < np; i++) { const hyp_t *h = &o->h[o->bare[part[i].idx]]; if (h->n >= min_len) { rk[nr].s = 0.9 * part[i].s / 100.0; rk[nr].idx = o->bare[part[i].idx]; rk[nr].seq = nr; nr++; } }
        qsort(rk, (size_t)nr, sizeof(rank_t), rank_cmp);
        unsigned char *seen = (unsigned char *)calloc((size_t)o->nh, 1); int *per = (int *)calloc((size_t)o->nmean + 1, sizeof(int));
        for (int i = 0; i < nr && nsl < K; i++) {
            int idx = rk[i].idx; if (seen[idx]) continue;
            int m = o->h[idx].meaning; if (per[m] >= 2) continue;
            per[m]++; seen[idx] = 1; sl[nsl++] = idx;
        }
        free(full); free(part); free(rk); free(seen); free(per);
    }
    double t_sl = now_ms();
    /* 3. 採点 (候補 + 埋め込み文)。埋め込み: 候補を自由認識の最も似た区間に置き換えた文 (Recognizer._embedded) */
    int nt = nsl, ntexts = nsl; int *emb_of = (int *)malloc(sizeof(int) * (size_t)(nsl + 1));
    uint32_t **tx = (uint32_t **)malloc(sizeof(uint32_t *) * (size_t)(2 * nsl + 1)); int *txn = (int *)malloc(sizeof(int) * (size_t)(2 * nsl + 1));
    for (int i = 0; i < nsl; i++) { tx[i] = o->h[sl[i]].k; txn[i] = o->h[sl[i]].n; }
    int nfm = n_morae(fk, nfk), nemb = 0;
    uint32_t *embbuf = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(nsl * (nfk + maxlen + 4) + 16)); size_t eo = 0;
    if (P[P_EMBED] > 0) {
        for (int i = 0; i < nsl; i++) {
            const hyp_t *h = &o->h[sl[i]];
            if (!(h->flags & 1)) continue;
            if (h->n == nfk && !memcmp(h->k, fk, sizeof(uint32_t) * (size_t)nfk)) continue;
            int nc = n_morae(h->k, h->n);
            if (nc < (int)P[P_EMB_MORAE] || nc < P[P_EMB_RATIO] * nfm || nfm - nc > (int)P[P_EMB_RESID]) continue;
            align_t al = partial_align(h->k, h->n, fk, nfk, 0.0, row);
            if (al.score < 0 || al.score < 55) continue;
            uint32_t *e = embbuf + eo; int ne = 0;
            memcpy(e, fk, sizeof(uint32_t) * (size_t)al.ds); ne += al.ds;
            memcpy(e + ne, h->k, sizeof(uint32_t) * (size_t)h->n); ne += h->n;
            memcpy(e + ne, fk + al.de, sizeof(uint32_t) * (size_t)(nfk - al.de)); ne += nfk - al.de;
            if (ne == h->n && !memcmp(e, h->k, sizeof(uint32_t) * (size_t)ne)) continue;
            emb_of[nemb++] = i; tx[ntexts] = e; txn[ntexts] = ne; ntexts++; eo += (size_t)ne;
        }
    }
    int *tokb = (int *)malloc(sizeof(int) * (size_t)(4 * (nfk + maxlen) + 64));
    double *sc = (double *)malloc(sizeof(double) * (size_t)ntexts), *ln = (double *)malloc(sizeof(double) * (size_t)ntexts);
    free(dpb); dpb = (double *)malloc(sizeof(double) * (size_t)(4 * (nfk + maxlen + 8) + 64));
    for (int i = 0; i < ntexts; i++) { int L = tok_encode(&o->tok, tx[i], txn[i], tokb); ln[i] = L; sc[i] = ctc_score(lp, T, V, tokb, L, dpb); }
    for (int j = 0; j < nemb; j++) {
        int i = emb_of[j]; double ne = ln[nt + j], borrowed = ne - ln[i] > 0 ? ne - ln[i] : 0;
        double se = sc[nt + j] - P[P_EMB_PEN] * borrowed * P[P_T];
        double zb = cal_logit(P, sc[i], ln[i], n_free), ze = cal_logit(P, se, ne, n_free);
        if (ze > zb) { sc[i] = se; ln[i] = ne; }
    }
    double t_sc = now_ms();
    /* 4. 校正 → [候補..., 該当なし] の確率 */
    double *z = (double *)malloc(sizeof(double) * (size_t)(nsl + 1)), zmax = -INFINITY, zs = 0;
    for (int i = 0; i < nsl; i++) { z[i] = cal_logit(P, sc[i], ln[i], n_free); if (z[i] > zmax) zmax = z[i]; }
    z[nsl] = (free_lp - P[P_B0]) / (P[P_T] > 1e-3 ? P[P_T] : 1e-3); if (z[nsl] > zmax) zmax = z[nsl];
    for (int i = 0; i <= nsl; i++) { z[i] = exp(z[i] - zmax); zs += z[i]; }
    for (int i = 0; i <= nsl; i++) z[i] /= zs;
    double none_p = z[nsl];
    bput(b, ",\"cands\":[");
    for (int i = 0; i < nsl; i++) bput(b, "%s[%d,%.6g,%.4f,%d]", i ? "," : "", sl[i], z[i], sc[i], (int)ln[i]);
    bput(b, "]");
    /* 5. 意味ごとに合算 (Recognizer.recognize)。順位は確率降順・同点は初出順 */
    int *mu = (int *)malloc(sizeof(int) * (size_t)(nsl + 1)), *mbest = (int *)malloc(sizeof(int) * (size_t)(nsl + 1)), nm = 0;
    double *mp = (double *)malloc(sizeof(double) * (size_t)(nsl + 1)), *ms = (double *)malloc(sizeof(double) * (size_t)(nsl + 1));
    for (int i = 0; i < nsl; i++) {
        int m = o->h[sl[i]].meaning, q = 0; while (q < nm && mu[q] != m) q++;
        if (q == nm) { mu[nm] = m; mp[nm] = z[i]; ms[nm] = sc[i]; mbest[nm] = sl[i]; nm++; }
        else { mp[q] += z[i]; if (sc[i] > ms[q]) { ms[q] = sc[i]; mbest[q] = sl[i]; } }
    }
    rank_t *mr = (rank_t *)malloc(sizeof(rank_t) * (size_t)(nm + 1));
    for (int q = 0; q < nm; q++) { mr[q].s = mp[q]; mr[q].idx = q; mr[q].seq = q; }
    qsort(mr, (size_t)nm, sizeof(rank_t), rank_cmp);
    bput(b, ",\"meanings\":[");
    for (int r = 0; r < nm; r++) { int q = mr[r].idx; bput(b, "%s[%d,%.6g,%.4f]", r ? "," : "", mbest[q], mp[q], ms[q]); }
    bput(b, "],\"none_prob\":%.6g", none_p);
    /* 6. 判断 (openvons.core.decision.decide) */
    const char *act = "none"; char why[160] = "no candidates";
    if (nm) {
        int q = mr[0].idx; double pt = mp[q]; const hyp_t *h = &o->h[mbest[q]];
        int confirmable = (h->flags & 2) != 0, positive = (h->flags & 4) != 0;
        if (none_p > pt) { act = "none"; snprintf(why, sizeof why, "該当なしが最尤 (%.2f)", none_p); }
        else if (!confirmable) {
            double need = positive ? P[P_ANS_YES] : P[P_ANS_NO];
            if (pt >= need) { act = "execute"; snprintf(why, sizeof why, "確認への返事 p=%.2f >= %g", pt, need); } else { act = "reject"; snprintf(why, sizeof why, "確信度不足"); }
        } else if (h->risk == 2) {
            if (pt >= P[P_CONF]) { act = "confirm"; snprintf(why, sizeof why, "危険度 high は常に確認"); } else { act = "reject"; snprintf(why, sizeof why, "確信度不足"); }
        } else {
            double rest = 1.0 - pt - none_p; if (rest < 0) rest = 0;
            int clear = pt >= P[P_CLEAR_MIN] && none_p <= P[P_CLEAR_NONE] && rest <= pt / P[P_CLEAR_RATIO];
            if (h->risk == 1) {
                if (pt >= P[P_EXEC_MED]) { act = "execute"; snprintf(why, sizeof why, "p=%.2f >= %g", pt, P[P_EXEC_MED]); }
                else if (pt >= P[P_CONF]) { act = "confirm"; snprintf(why, sizeof why, "p=%.2f (medium)", pt); }
                else { act = "reject"; snprintf(why, sizeof why, "確信度不足"); }
            } else if (pt >= P[P_EXEC]) { act = "execute"; snprintf(why, sizeof why, "p=%.2f >= %g", pt, P[P_EXEC]); }
            else if (clear) { act = "execute"; snprintf(why, sizeof why, "p=%.2f で競合なし (他候補 %.2f / 該当なし %.2f)", pt, rest, none_p); }
            else if (pt >= P[P_CONF]) { act = "confirm"; snprintf(why, sizeof why, "p=%.2f in [%g,%g)", pt, P[P_CONF], P[P_EXEC]); }
            else { act = "reject"; snprintf(why, sizeof why, "p=%.2f < %g", pt, P[P_CONF]); }
        }
    }
    double t_end = now_ms();
    bput(b, ",\"action\":\"%s\",\"reason\":\"%s\"", act, why);
    bput(b, ",\"timings_ms\":{\"encode\":%.1f,\"transcribe\":%.1f,\"shortlist\":%.1f,\"score\":%.1f,\"total\":%.1f}}",
         t_enc, t_tr - t0, t_sl - t_tr, t_sc - t_sl, t_enc + t_end - t0);
    free(ids); free(dpb); free(raw); free(fk); free(sl); free(row); free(emb_of); free(tx); free(txn); free(embbuf); free(tokb); free(sc); free(ln); free(z);
    free(mu); free(mbest); free(mp); free(ms); free(mr);
    return b->p;
}

API const char *ovk_route_logprobs(void *p, const float *lp, int T) { return route_lp((ovk_t *)p, lp, T, 0.0); }

static void log_softmax_rows(float *x, int T, int V) {
    for (int t = 0; t < T; t++) {
        float *r = x + (size_t)t * V, m = r[0]; for (int v = 1; v < V; v++) if (r[v] > m) m = r[v];
        double s = 0; for (int v = 0; v < V; v++) s += exp((double)(r[v] - m));
        float ls = (float)log(s); for (int v = 0; v < V; v++) r[v] = r[v] - m - ls;
    }
}
/* 音声 → logits (komimi 全文脈) → log-softmax → 振り分け。0.6 秒未満は無音で伸ばす (KomimiASR.MIN_SEC と同じ) */
API const char *ovk_route(void *p, const float *pcm, int n) {
    ovk_t *o = (ovk_t *)p; double t0 = now_ms();
    int nn = n < 9600 ? 9600 : n; const float *x = pcm; float *pad = NULL;
    if (nn != n) { pad = (float *)calloc((size_t)nn, sizeof(float)); memcpy(pad, pcm, sizeof(float) * (size_t)n); x = pad; }
    int maxf = nn / (o->m.hop * 4) + 4, V = o->m.vocab;
    if (maxf * V > o->lp_cap) { free(o->lp); o->lp_cap = maxf * V; o->lp = (float *)malloc(sizeof(float) * (size_t)o->lp_cap); }
    int ids[4096], T = 0;
    km_recognize_ex(&o->m, x, nn, ids, 4096, &T, o->lp, maxf);
    if (T > maxf) T = maxf;
    log_softmax_rows(o->lp, T, V);
    free(pad);
    return route_lp(o, o->lp, T, now_ms() - t0);
}
/* 自由認識だけ (表示用) */
API const char *ovk_transcribe(void *p, const float *pcm, int n) {
    ovk_t *o = (ovk_t *)p; int ids[4096], T = 0, k = km_recognize(&o->m, pcm, n, ids, 4096, &T);
    o->out.n = 0; char tmp[8192]; km_detok(&o->m, ids, k < 0 ? 0 : k, tmp, sizeof tmp);
    bput(&o->out, "%s", tmp); return o->out.p;
}
API void ovk_free(void *p) {
    ovk_t *o = (ovk_t *)p; if (!o) return;
    free(o->h); free(o->pool); free(o->bare); free(o->tok.score); free(o->tok.tab); free(o->out.p); free(o->lp); free(o->mbuf); free(o);
}
