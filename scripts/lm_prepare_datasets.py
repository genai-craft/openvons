"""Download public datasets and convert them into the decision JSONL format.

Benchmarks (spec §8):
  A intent      : MASSIVE scenario (18 classes) en / ja          -> choice
  A' intent     : MASSIVE intent   (60 classes) en                -> choice (n > 26)
  B moderation  : tweet_eval offensive (binary)                   -> noul
  C score       : Yelp review full (5 stars)                      -> score
  D tool routing: glaive-function-calling-v2 (variable tool set)  -> choice (unseen tools at test)

Output: data/store/processed/<task>/{train,valid,test}.jsonl
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/decision_model/hf_home")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download  # noqa: E402
from openvons.core.primitives import Question, Option  # noqa: E402
from openvons.core.formats import Sample, write_jsonl  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data/store/processed"
random.seed(0)

SCENARIO_DESC = {
    "alarm": "setting, changing or querying alarms",
    "audio": "device volume and audio settings",
    "calendar": "calendar events, schedules and reminders",
    "cooking": "recipes and cooking instructions",
    "datetime": "current date, time or time zones",
    "email": "reading, sending or querying email",
    "general": "greetings, jokes, chit-chat and other general requests",
    "iot": "smart home devices: lights, plugs, cleaner, coffee machine",
    "lists": "creating, editing or querying lists",
    "music": "music playback preferences (likes, settings)",
    "news": "news articles and headlines",
    "play": "playing music, radio, podcasts, audiobooks or games",
    "qa": "factual questions, definitions, currency, stock, math",
    "recommendation": "recommendations for events, movies, restaurants, locations",
    "social": "social media posts and queries",
    "takeaway": "food delivery and takeaway orders",
    "transport": "taxi, tickets, traffic and public transport",
    "weather": "weather forecasts and conditions",
}
SCENARIO_DESC_JA = {
    "alarm": "アラームの設定・変更・確認", "audio": "音量やオーディオ設定", "calendar": "予定・スケジュール・リマインダー",
    "cooking": "レシピや料理の手順", "datetime": "現在の日時やタイムゾーン", "email": "メールの確認・送信・検索",
    "general": "挨拶・ジョーク・雑談などの一般的な要求", "iot": "スマートホーム機器 (照明・プラグ・掃除機・コーヒーメーカー)",
    "lists": "リストの作成・編集・確認", "music": "音楽再生の好み・設定", "news": "ニュース記事や見出し",
    "play": "音楽・ラジオ・ポッドキャスト・オーディオブック・ゲームの再生", "qa": "事実に関する質問・定義・為替・株価・計算",
    "recommendation": "イベント・映画・レストラン・場所のおすすめ", "social": "SNS への投稿や確認",
    "takeaway": "フードデリバリーやテイクアウトの注文", "transport": "タクシー・チケット・交通情報・公共交通機関",
    "weather": "天気予報や気象状況",
}


def load_json_gz(repo: str, path: str) -> list[dict]:
    fp = hf_hub_download(repo, path, repo_type="dataset")
    rows = []
    with gzip.open(fp, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump(task: str, split: str, samples: list[Sample]):
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(str(d / f"{split}.jsonl"), samples)
    print(f"  {task}/{split}: {n}")


# --------------------------------------------------------------------- MASSIVE
def prep_massive_scenario():
    for lang in ["en", "ja"]:
        desc = SCENARIO_DESC if lang == "en" else SCENARIO_DESC_JA
        qtext = ("What is the scenario (domain) of this user utterance to a voice assistant?" if lang == "en"
                 else "この音声アシスタントへの発話はどのシナリオ(ドメイン)に属するか?")
        labels = sorted(desc)
        options = [Option(k, desc[k]) for k in labels]
        for split, hfsplit in [("train", "train"), ("valid", "validation"), ("test", "test")]:
            rows = load_json_gz("mteb/amazon_massive_scenario", f"{hfsplit}/{lang}.json.gz")
            samples = []
            for i, r in enumerate(rows):
                lab = r["label"] if isinstance(r["label"], str) else r.get("label_text")
                if lab not in desc:
                    continue
                samples.append(Sample(f"massive_scenario_{lang}_{split}_{i}", r["text"],
                                      Question("choice", qtext, options), label=labels.index(lab)))
            dump(f"massive_scenario_{lang}", split, samples)


def prep_massive_intent():
    lang = "en"
    for split, hfsplit in [("train", "train"), ("valid", "validation"), ("test", "test")]:
        rows = load_json_gz("mteb/amazon_massive_intent", f"{hfsplit}/{lang}.json.gz")
        if split == "train":
            labels = sorted({r["label"] for r in rows})
            prep_massive_intent.labels = labels
        labels = prep_massive_intent.labels
        options = [Option(k, k.replace("_", " ")) for k in labels]
        q = Question("choice", "What is the intent of this user utterance to a voice assistant?", options)
        samples = [Sample(f"massive_intent_{lang}_{split}_{i}", r["text"], q, label=labels.index(r["label"]))
                   for i, r in enumerate(rows) if r["label"] in labels]
        dump(f"massive_intent_{lang}", split, samples)


# ------------------------------------------------------------------ tweet_eval
def prep_tweet_offensive():
    import pandas as pd
    q = Question("noul", "Is this tweet offensive?", [Option("true", "offensive"), Option("false", "not offensive")])
    for split, hfsplit in [("train", "train"), ("valid", "validation"), ("test", "test")]:
        fp = hf_hub_download("cardiffnlp/tweet_eval", f"offensive/{hfsplit}-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(fp)
        samples = [Sample(f"tweet_offensive_{split}_{i}", str(r.text), q, label=0 if int(r.label) == 1 else 1)
                   for i, r in enumerate(df.itertuples())]
        dump("tweet_offensive", split, samples)


# ------------------------------------------------------------------------ Yelp
def prep_yelp(n_train=20000, n_valid=2000, n_test=5000):
    import pandas as pd
    from huggingface_hub import list_repo_files
    files = list_repo_files("Yelp/yelp_review_full", repo_type="dataset")
    levels = ["1 star (very negative)", "2 stars (negative)", "3 stars (neutral / mixed)",
              "4 stars (positive)", "5 stars (very positive)"]
    q = Question("score", "How many stars did the reviewer give? Rate the sentiment of this review.",
                 [Option(str(i), lv) for i, lv in enumerate(levels)])

    def load(prefix):
        fs = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
        return pd.concat([pd.read_parquet(hf_hub_download("Yelp/yelp_review_full", f, repo_type="dataset")) for f in fs])

    tr = load("yelp_review_full/train").sample(n=n_train + n_valid, random_state=0)
    te = load("yelp_review_full/test").sample(n=n_test, random_state=0)

    def conv(df, split):
        return [Sample(f"yelp_{split}_{i}", str(r.text)[:2000], q, label=int(r.label)) for i, r in enumerate(df.itertuples())]

    dump("yelp_score", "train", conv(tr.iloc[:n_train], "train"))
    dump("yelp_score", "valid", conv(tr.iloc[n_train:], "valid"))
    dump("yelp_score", "test", conv(te, "test"))


# ---------------------------------------------------------------------- glaive
def prep_glaive(n_train=20000, n_valid=2000, n_test=5000):
    fp = hf_hub_download("glaiveai/glaive-function-calling-v2", "glaive-function-calling-v2.json", repo_type="dataset")
    data = json.load(open(fp))
    random.shuffle(data)
    fn_re = re.compile(r"\{.*\}", re.S)
    out = []
    for i, rec in enumerate(data):
        sys_txt = rec["system"]
        m = sys_txt.find("[") if "functions. Use them if required -" in sys_txt else -1
        # functions are given as one or more JSON objects after the header line
        header_end = sys_txt.find("-\n")
        if header_end < 0:
            continue
        body = sys_txt[header_end + 2:].strip()
        funcs = []
        # try list first, then concatenated objects
        try:
            parsed = json.loads(body)
            funcs = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            depth, start = 0, None
            for j, ch in enumerate(body):
                if ch == "{":
                    if depth == 0:
                        start = j
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        try:
                            funcs.append(json.loads(body[start:j + 1]))
                        except Exception:
                            pass
                        start = None
        funcs = [f for f in funcs if isinstance(f, dict) and "name" in f]
        seen, uniq = set(), []
        for f in funcs:
            if f["name"] not in seen and f["name"] != "none":
                seen.add(f["name"]); uniq.append(f)
        funcs = uniq
        if not funcs:
            continue
        chat = rec["chat"]
        if not chat.startswith("USER:"):
            continue
        a = chat.find("ASSISTANT:")
        if a < 0:
            continue
        user = chat[len("USER:"):a].strip()
        rest = chat[a + len("ASSISTANT:"):].strip()
        nxt = rest.find("USER:")
        first_reply = rest if nxt < 0 else rest[:nxt]
        if "<functioncall>" in first_reply:
            m = re.search(r'"name"\s*:\s*"([^"]+)"', first_reply.split("<functioncall>", 1)[1])
            if not m:
                continue
            target = m.group(1)
            if target not in [f["name"] for f in funcs]:
                continue
        else:
            target = "none"
        options = [Option(f["name"], str(f.get("description", ""))[:200]) for f in funcs] + [Option("none", "no tool call is needed; answer directly")]
        q = Question("choice", "Which tool should the assistant call to handle the user's latest message?", options)
        # split by hash of the *first* tool name so that test tools are unseen during training
        h = int(hashlib.md5(funcs[0]["name"].encode()).hexdigest(), 16) % 100
        split = "test" if h < 15 else ("valid" if h < 22 else "train")
        out.append((split, Sample(f"glaive_{i}", f"User: {user}", q, label=q.ids.index(target),
                                  meta={"n_tools": len(funcs)})))
    by = {"train": [], "valid": [], "test": []}
    for s, smp in out:
        by[s].append(smp)
    dump("glaive_tools", "train", by["train"][:n_train])
    dump("glaive_tools", "valid", by["valid"][:n_valid])
    dump("glaive_tools", "test", by["test"][:n_test])
    tr_tools = {o.id for s in by["train"][:n_train] for o in s.question.options}
    te_tools = {o.id for s in by["test"][:n_test] for o in s.question.options}
    print(f"  glaive: train tools={len(tr_tools)} test tools={len(te_tools)} overlap={len(tr_tools & te_tools)}")
    from collections import Counter
    print("  glaive label none-rate test:", Counter(s.question.ids[s.label] == 'none' for s in by['test'][:n_test]))


if __name__ == "__main__":
    which = sys.argv[1:] or ["massive_scenario", "massive_intent", "tweet", "yelp", "glaive"]
    for w in which:
        print("==", w)
        {"massive_scenario": prep_massive_scenario, "massive_intent": prep_massive_intent,
         "tweet": prep_tweet_offensive, "yelp": prep_yelp, "glaive": prep_glaive}[w]()
