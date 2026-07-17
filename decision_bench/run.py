#!/usr/bin/env python3
"""Decision-reintroduction benchmark v2 — cross-model (OpenAI-compatible) runner.

Same methodology as the keyless Claude runner (keyless_bench.mjs); this path is
the OPTIONAL cross-model leg (e.g. GPT via the Codex proxy or a real key), so a
gain that holds on BOTH Claude and GPT is model-independent.

Methodology (v2, post adversarial review):
  * 3 arms on `behavioral` items: A (no brief), B (real why() brief), B' (decoy
    generic brief). memory-attributable effect = B_R - B'_R.  general_knowledge /
    recall items run A/B only and are reported separately (never in the headline).
  * BLIND judge: sees only the bare actions, shuffled (deterministically) as
    Option 1/2; mapped back to R/W after. Judge should be a DIFFERENT model
    (--judge-model); it defaults to gpt-4o-mini-ish if you set one, else a warning.
  * R-ADOPTION metric (recommends==R), not W-avoidance; both/neither/unparsed are a
    NO-COMMIT bucket, never a win.
  * VERIFIED citation: the item's evidence_token must appear verbatim in the answer.
  * Negation-aware keyword cross-check, surfaced with judge-disagreement flags.

    OPENAI_BASE_URL=... OPENAI_API_KEY=... \
        python run.py --model gpt-5.5 --judge-model gpt-4o-mini --trials 4

Reads corpus_v2.json + briefs.json from disk (no workflow arg limits).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = (
    "You are a senior engineer working on Kumiho Memory, a graph-native long-term "
    "memory system for AI agents (Python, async, Neo4j + hybrid search). Answer the "
    "design question with a concrete recommendation in 3-6 sentences. Commit to ONE "
    "of the two options; do not hedge between both."
)

JUDGE_SYSTEM = (
    "You are a strict, neutral evaluator. Decide which of the two options the "
    "engineer's answer recommends, and whether it cites a specific measurement, "
    "benchmark number, or past incident. Reply with ONLY compact JSON: "
    '{"chosen_option": "1"|"2"|"both"|"neither", "cites_specific_evidence": true|false}. '
    "Judge only what the answer says."
)


def client_from_env():
    from openai import OpenAI

    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        api_key=os.environ.get("OPENAI_API_KEY") or "sk-none",
    )


def chat(client, model, system, user, temperature):
    import time
    kwargs = dict(model=model, messages=[
        {"role": "system", "content": system}, {"role": "user", "content": user}])
    last = None
    for attempt in range(3):  # survive transient API errors on a long paid run
        try:
            try:
                r = client.chat.completions.create(temperature=temperature, **kwargs)
            except Exception as exc:  # only drop temperature on an explicit rejection
                if "temperature" not in str(exc).lower():
                    raise
                print(f"  (note: {model} rejected temperature={temperature}; using default)", file=sys.stderr)
                r = client.chat.completions.create(**kwargs)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last = e
            print(f"  (retry {attempt + 1}/3 after error: {str(e)[:120]})", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise last


def with_brief(task, brief):
    if not brief:
        return task
    return "Relevant prior decisions from this project's Decision Memory:\n" + brief + "\n\n---\n" + task


def hash01(s):
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0x7FFFFFFF
    return h % 2


_NEG = re.compile(r"(not|n't|never|avoid|without|instead of|rather than|no)\b[^.]{0,18}$")


def hits_neg_aware(text, signals):
    t = (text or "").lower()
    n = 0
    for s in signals or []:
        sl = str(s).lower()
        start = 0
        while True:
            idx = t.find(sl, start)
            if idx == -1:
                break
            if not _NEG.search(t[max(0, idx - 24):idx]):
                n += 1
            start = idx + len(sl)
    return n


def judge(client, model, item, answer, right_is_one):
    opt1 = item["right_action"] if right_is_one else item["wrong_action"]
    opt2 = item["wrong_action"] if right_is_one else item["right_action"]
    user = (
        f"DESIGN QUESTION:\n{item['task']}\n\n"
        f"Option 1: {opt1}\nOption 2: {opt2}\n\n"
        f"ENGINEER'S ANSWER:\n{answer}\n\n"
        "Which option (1 or 2, or both/neither) does the answer recommend, and cites_specific_evidence? JSON only."
    )
    raw = chat(client, model, JUDGE_SYSTEM, user, 0.0)
    m = re.search(r"\{.*\}", raw, re.S)
    obj = {}
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = {}
    chosen = str(obj.get("chosen_option", "")).lower()
    if chosen == "both":
        rec = "both"
    elif chosen == "1":
        rec = "R" if right_is_one else "W"
    elif chosen == "2":
        rec = "W" if right_is_one else "R"
    else:
        rec = "neither"
    return rec, bool(obj.get("cites_specific_evidence", False))


def arms_for(item):
    return ["A", "B", "Bprime"] if item["bucket"] == "behavioral" else ["A", "B"]


def brief_for(item, arm, briefs):
    if arm == "B":
        return briefs.get(item["id"], {}).get("memory_brief", "")
    if arm == "Bprime":
        return item.get("decoy_brief", "")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DECISION_BENCH_MODEL", "gpt-4o"))
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--agent-temp", type=float, default=0.7)
    ap.add_argument("--only", default="")
    ap.add_argument("--corpus", default="corpus_v2.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.judge_model:
        print("WARNING: --judge-model unset; judge == agent model (self-judging bias). "
              "Pass a different --judge-model for the headline run.", file=sys.stderr)
    judge_model = args.judge_model or args.model

    with open(os.path.join(HERE, args.corpus), encoding="utf-8") as fh:
        corpus = json.load(fh)
    briefs = {}
    bp = os.path.join(HERE, "briefs.json")
    if os.path.exists(bp):
        with open(bp, encoding="utf-8") as fh:
            briefs = json.load(fh)
    elif not args.dry_run:
        print("WARNING: briefs.json missing — arm B has no memory.", file=sys.stderr)

    items = corpus["items"]
    if args.only:
        want = set(args.only.split(","))
        items = [i for i in items if i["id"] in want]

    if args.dry_run:
        for it in items:
            for arm in arms_for(it):
                print(f"== {it['id']} [{arm}] ==\n{with_brief(it['task'], brief_for(it, arm, briefs))}\n")
        return 0

    client = client_from_env()
    trials = []  # flat list of graded records
    for it in items:
        for arm in arms_for(it):
            brief = brief_for(it, arm, briefs)
            for t in range(args.trials):
                ans = chat(client, args.model, SYSTEM, with_brief(it["task"], brief), args.agent_temp)
                rio = hash01(it["id"] + arm + str(t)) == 0
                rec, cites = judge(client, judge_model, it, ans, rio)
                tok = (it.get("evidence_token") or "").lower()
                vcite = (tok in ans.lower()) if tok else None
                rH = hits_neg_aware(ans, it.get("right_signals"))
                wH = hits_neg_aware(ans, it.get("wrong_signals"))
                kw = "R" if rH > wH else ("W" if wH > rH else "tie")
                trials.append({"id": it["id"], "bucket": it["bucket"], "arm": arm, "trial": t,
                               "recommends": rec, "cites": cites, "verified_cite": vcite,
                               "kw": kw, "kw_disagree": kw in ("R", "W") and rec in ("R", "W") and kw != rec})
        print(f"  done {it['id']}")

    report = render(corpus, items, trials, args, judge_model)
    with open(os.path.join(HERE, "report_v2.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(os.path.join(HERE, "raw_results_v2.json"), "w", encoding="utf-8") as fh:
        json.dump(trials, fh, ensure_ascii=False, indent=2)
    print("\n" + report)
    return 0


def render(corpus, items, trials, args, judge_model):
    def sel(iid, arm):
        return [r for r in trials if r["id"] == iid and r["arm"] == arm]

    def rate(arr, pred):
        return (sum(1 for r in arr if pred(r)) / len(arr)) if arr else 0.0

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else 0.0

    isR = lambda r: r["recommends"] == "R"
    per = []
    for it in items:
        A, B, Bp = sel(it["id"], "A"), sel(it["id"], "B"), sel(it["id"], "Bprime")
        vc = [r for r in B if r["verified_cite"] is not None]
        per.append({
            "id": it["id"], "bucket": it["bucket"],
            "a_R": rate(A, isR), "b_R": rate(B, isR),
            "bp_R": rate(Bp, isR) if it["bucket"] == "behavioral" else None,
            "b_nocommit": rate(B, lambda r: r["recommends"] in ("both", "neither")),
            "b_vcite": (rate(vc, lambda r: r["verified_cite"]) if it.get("evidence_token") else None),
            "already_knew": rate(A, isR) >= 0.5,
        })
    beh = [p for p in per if p["bucket"] == "behavioral"]

    L = ["# Decision-Reintroduction Benchmark v2 — report", ""]
    L.append(f"- agent model: `{args.model}`  judge: `{judge_model}`  trials/arm: {args.trials}")
    L.append(f"- behavioral items: {len(beh)}  (headline); general_knowledge / recall reported separately")
    L.append("")
    L.append("## Headline — behavioral items (R = evidence-backed choice)")
    L.append("")
    L.append("| item | A R-rate | B' (decoy) R | B (memory) R | **content effect B−B'** | verified cite |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for p in beh:
        vc = f"{p['b_vcite']:.2f}" if p["b_vcite"] is not None else "—"
        ce = (p["b_R"] - p["bp_R"])
        L.append(f"| `{p['id']}` | {p['a_R']:.2f} | {p['bp_R']:.2f} | {p['b_R']:.2f} | {ce:+.2f} | {vc} |")
    mA, mBp, mB = mean([p["a_R"] for p in beh]), mean([p["bp_R"] for p in beh]), mean([p["b_R"] for p in beh])
    L.append("")
    L.append(f"- baseline (A) R-rate: **{mA:.2f}**")
    L.append(f"- decoy (B′) R-rate: **{mBp:.2f}**   → authority effect (B′−A): {mBp - mA:+.2f}")
    L.append(f"- memory (B) R-rate: **{mB:.2f}**")
    L.append(f"- **content effect (B − B′), memory-attributable: {mB - mBp:+.2f}**")
    L.append(f"- raw gain (B − A): {mB - mA:+.2f}")
    knew = [p["id"] for p in beh if p["already_knew"]]
    if knew:
        L.append(f"- model already knew (A R≥0.5): {', '.join('`'+i+'`' for i in knew)}")
    L.append(f"- keyword/judge disagreements: {sum(1 for r in trials if r['kw_disagree'])}")
    L.append("")
    for bucket, title in [("general_knowledge", "General-knowledge (a-priori, NOT headline)"),
                          ("recall", "Internal-state recall (NOT behavioral)")]:
        br = [p for p in per if p["bucket"] == bucket]
        if not br:
            continue
        L.append(f"## {title}")
        L.append("")
        L.append("| item | A R-rate | B (memory) R | gain B−A |")
        L.append("|---|--:|--:|--:|")
        for p in br:
            L.append(f"| `{p['id']}` | {p['a_R']:.2f} | {p['b_R']:.2f} | {p['b_R'] - p['a_R']:+.2f} |")
        L.append("")
    L.append("## How to read this")
    L.append("- **content effect (B − B′)** is the memory-attributable number: how much more the "
             "agent adopts the evidence-backed choice with the REAL brief vs a generic decoy brief. "
             "Raw gain (B − A) also credits 'any authoritative in-context text.'")
    L.append("- R-adoption (not W-avoidance) is the metric; both/neither don't count as wins.")
    L.append("- verified cite = the item's measured token appeared verbatim (deterministic).")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
