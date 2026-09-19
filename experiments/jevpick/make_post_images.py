"""X / note 投稿用の画像 (1200x675, LP と同じ暗色) を docs/jevpick/img/ に書き出す。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "docs/jevpick/img"
OUT.mkdir(parents=True, exist_ok=True)
BG, PANEL, LINE, FG, MUTED = "#0b0e13", "#141b24", "#222c38", "#e8eef5", "#8fa0b3"
BLUE, ORANGE, GRAY = "#3987e5", "#d95926", "#5c6b7a"   # 検証済 (dark surface): JevPick / 他手法 / 何もしない
W, H, DPI = 1200, 675, 100
for f in font_manager.findSystemFonts():
    if "ipagp" in f.lower() or "IPAPGothic" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["IPAPGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def canvas():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    return fig


def brand(fig, y=0.06):
    fig.text(0.05, y, "openvons", color=FG, fontsize=15, fontweight="bold", va="center")
    fig.patches.append(FancyBboxPatch((0.028, y - 0.013), 0.014, 0.026, boxstyle="round,pad=0,rounding_size=0.004",
                                      transform=fig.transFigure, facecolor=BLUE, edgecolor="none"))
    fig.text(0.95, y, "openvons.com/jevpick", color=MUTED, fontsize=13, ha="right", va="center")


def rounded_bar(ax, y, x0, x1, h, color):
    """4px 丸の先端、根元は角。"""
    r = min(h / 2, 0.04 * (x1 - x0) if x1 > x0 else 0)
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=color, edgecolor="none", mutation_aspect=1))
    ax.add_patch(plt.Rectangle((x0, y - h / 2), min(r, x1 - x0), h, facecolor=color, edgecolor="none"))


def hbar_chart(ax, rows, xmax, unit="倍", highlight=BLUE, other=ORANGE, base=GRAY):
    """rows: [(label, value, kind)] kind in {jev, other, base}"""
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.set_yticks([]); ax.set_xticks([])
    for i, (label, v, kind) in enumerate(rows):
        color = {"jev": highlight, "other": other, "base": base}[kind]
        rounded_bar(ax, i, 0, v, 0.36, color)
        ax.text(-0.015 * xmax, i, label, color=FG if kind == "jev" else MUTED, fontsize=15, ha="right", va="center",
                fontweight="bold" if kind == "jev" else "normal")
        ax.text(v + 0.018 * xmax, i, f"{v:g}{unit}", color=FG, fontsize=16, va="center", fontweight="bold" if kind == "jev" else "normal")
    if unit == "倍":
        ax.axvline(1.0, color=LINE, lw=1)
    else:
        ax.axvline(0, color=LINE, lw=1)


# ---------- 1. ヒーロー: 数字カード ----------
fig = canvas()
fig.text(0.05, 0.86, "JevPick", color=FG, fontsize=44, fontweight="bold", va="center")
fig.text(0.05, 0.75, "AIに「選ばせる」を、AIが文章を書く速さそのものに使う", color=MUTED, fontsize=19, va="center")
cards = [("3.2〜4.8倍", "ツール呼び出しの生成速度。\n予測モデルなし、\n出力は完全一致"),
         ("90%", "ツール定義から作った\n候補メニューに正解の\n続きが入っている割合"),
         ("64% → 88%", "メニューから正解を\n選ぶ精度。\n単純ルール → JevPick"),
         ("5.4倍 → 1.0倍", "流行りの先読み手法が\n16k語の長い文脈で\n失う速さ")]
for k, (big, small) in enumerate(cards):
    x = 0.05 + k * 0.2275
    fig.patches.append(FancyBboxPatch((x, 0.16), 0.205, 0.48, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor=PANEL, edgecolor=LINE, lw=1))
    fig.text(x + 0.016, 0.54, big, color=BLUE, fontsize=27, fontweight="bold", va="center")
    fig.text(x + 0.016, 0.35, small, color=FG, fontsize=13, va="center", linespacing=1.7)
brand(fig)
fig.savefig(OUT / "01_hero.png", facecolor=BG); plt.close(fig)

# ---------- 2. 手法比較 (Qwen3-4B, Tool Call) ----------
fig = canvas()
fig.text(0.05, 0.9, "ツール呼び出しの生成速度 — 何もしない = 1.0倍", color=FG, fontsize=23, fontweight="bold", va="center")
fig.text(0.05, 0.83, "Qwen3-4B、同じ検算コードで方式だけを変えて実測。JevPick は別の予測モデルを学習していない。", color=MUTED, fontsize=14, va="center")
ax = fig.add_axes([0.30, 0.16, 0.62, 0.6])
hbar_chart(ax, [("何もしない", 1.0, "base"), ("DFlash (学習済み予測モデル)", 2.9, "other"),
                ("JevPick (候補16語)", 4.8, "jev"), ("JevPick (候補8語)", 3.8, "jev"), ("JevPick + DFlash 併用", 4.0, "jev")], 5.6)
fig.text(0.05, 0.1, "■ JevPick", color=BLUE, fontsize=13, va="center"); fig.text(0.16, 0.1, "■ 既存の先読み手法", color=ORANGE, fontsize=13, va="center")
fig.text(0.35, 0.1, "■ 何もしない", color=GRAY, fontsize=13, va="center")
fig.text(0.95, 0.1, "greedy・1リクエスト・RTX PRO 6000", color=MUTED, fontsize=12, ha="right", va="center")
brand(fig, y=0.04)
fig.savefig(OUT / "02_methods.png", facecolor=BG); plt.close(fig)

# ---------- 3. 長文脈で効果が消える ----------
fig = canvas()
fig.text(0.05, 0.9, "長い文脈では、先読みそのものが逆効果になる", color=FG, fontsize=23, fontweight="bold", va="center")
fig.text(0.05, 0.83, "vLLM 実測 (Qwen3-4B、ツール呼び出し)。300語 → 16k語の文脈で、何もしない=1.0倍に対する速さ。", color=MUTED, fontsize=14, va="center")
ax = fig.add_axes([0.10, 0.18, 0.84, 0.58]); ax.set_facecolor(BG)
for s in ax.spines.values(): s.set_visible(False)
groups = [("DFlash", 5.38, 0.99), ("n-gram (prompt lookup)", 1.94, 0.76), ("JevPick を「やらない判断」に使う", None, 1.00)]
ax.set_xlim(-0.5, len(groups) - 0.3); ax.set_ylim(0, 6.3); ax.set_xticks([]); ax.set_yticks([])
ax.axhline(1.0, color=LINE, lw=1); ax.text(2.68, 0.78, "─ 何もしない = 1.0倍", color=MUTED, fontsize=12, ha="right")
bw = 0.28
for i, (name, a, b) in enumerate(groups):
    if a is not None:
        ax.add_patch(FancyBboxPatch((i - bw - 0.02, 0), bw, a, boxstyle="round,pad=0,rounding_size=0.03", facecolor=ORANGE, edgecolor="none"))
        ax.text(i - bw / 2 - 0.02, a + 0.15, f"{a:.1f}倍", color=FG, fontsize=15, ha="center", fontweight="bold")
        ax.text(i - bw / 2 - 0.02, -0.35, "300語", color=MUTED, fontsize=12, ha="center")
    col = BLUE if "JevPick" in name else ORANGE
    ax.add_patch(FancyBboxPatch((i + 0.02, 0), bw, b, boxstyle="round,pad=0,rounding_size=0.03", facecolor=col, edgecolor="none"))
    ax.text(i + 0.02 + bw / 2, b + 0.15, f"{b:.2f}倍", color=FG, fontsize=15, ha="center", fontweight="bold")
    ax.text(i + 0.02 + bw / 2, -0.35, "16k語", color=MUTED, fontsize=12, ha="center")
    ax.text(i, -0.85, name, color=FG, fontsize=14, ha="center")
    if a is not None:
        ax.annotate("", xy=(i + 0.02 + bw / 2, b + 0.6), xytext=(i - bw / 2 - 0.02, a - 0.3),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2, connectionstyle="arc3,rad=-0.25"))
ax.text(2, 4.6, "常時先読みする方式は\n16kで 0.6〜0.8倍まで落ちる。\n「いま先読みすべきか」の判断が\n損失を止める。", color=FG, fontsize=14, ha="center", va="center", linespacing=1.6)
brand(fig, y=0.04)
fig.savefig(OUT / "03_long_context.png", facecolor=BG); plt.close(fig)

# ---------- 4. 選ぶ精度 ----------
fig = canvas()
fig.text(0.05, 0.9, "候補はある。あとは「選ぶ精度」の問題だった", color=FG, fontsize=23, fontweight="bold", va="center")
fig.text(0.05, 0.83, "ツール呼び出し、Qwen3.8-27B。候補メニューは学習なし (ツール定義と過去の出力から機械的に作る)。", color=MUTED, fontsize=14, va="center")
ax = fig.add_axes([0.36, 0.2, 0.58, 0.52])
hbar_chart(ax, [("メニューに正解の続きが入っている", 90, "base"), ("単純ルール (頻度順) で当てる", 64, "other"), ("JevPick で当てる", 88, "jev")], 105, unit="%")
fig.text(0.05, 0.1, "JevPick: パラメータ 500万、学習 1〜2分。bf16 で学習したものを FP8 / 4-bit のモデルにそのまま使っても低下は 0.5〜3pt。", color=MUTED, fontsize=12.5, va="center")
brand(fig, y=0.04)
fig.savefig(OUT / "04_accuracy.png", facecolor=BG); plt.close(fig)
print("->", OUT, sorted(p.name for p in OUT.glob("*.png")))
