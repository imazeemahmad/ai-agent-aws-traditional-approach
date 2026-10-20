"""Evaluation harness: score an implementation of the assistant against labelled cases (eval/cases.yaml).

Why: choosing between architectures (tool-choice agent, router, LangGraph, Strands...) should be decided by scores on the
same questions, not by opinion. Every check is deterministic (no LLM judge), so a score change means behaviour changed.

Per case, three checks must all pass:
  route     the agents that actually supplied sources match the expected route: exposure | price | both | none
  answer    every must_include is present (literal, case-insensitive; "re:" prefix = regex; a list = any of),
            no must_not_include is present, and refuse:true cases contain a refusal / "not available" statement
  grounded  every money amount, percentage and k/M figure in the answer appears in a retrieved passage (or the question)

Run:   python main.py --eval [eval/cases.yaml] [--eval-label NAME] [--eval-repeat N] [--eval-filter TEXT]
Diff:  python main.py --eval-compare eval/results/A.json eval/results/B.json

To score a different architecture, add a class with  ask(question, history) -> Result  to TARGETS and pass
--eval-target NAME. Nothing else in this file depends on how the answer was produced.
"""
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from . import chat
from .assistant import Assistant

ROUTE_OF_AGENT = {"main": "exposure", "price": "price"}  # source "agent" tag -> route name
REFUSAL = re.compile(
    r"couldn't find|could not find|can't answer|cannot answer|not available|no (relevant )?(price )?(data|information)|"
    r"do(es)? not contain|doesn't contain|don't have|do not have|not in the (knowledge base|sources)|unable to",
    re.I)
_NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)(?:\s?(million)\b|([kKmM])\b|(%))?", re.I)


@dataclass
class Result:
    """What one implementation produced for one question."""
    text: str                                     # everything the user saw: answer plus any sub-agent block
    agents: set = field(default_factory=set)      # {"main", "price"}: agents whose sources were used
    passages: list = field(default_factory=list)  # text of every passage retrieved while answering
    llm_calls: int = 0
    retrievals: int = 0
    latency_s: float = 0.0


class AssistantTarget:
    """The current implementation: Assistant (exposure KB + price sub-agent tool)."""

    def __init__(self, cfg, profile=None):
        self.assistant = Assistant(cfg, cfg["chat"]["default_model"], profile)
        self.passages, self.count = [], {"llm": 0, "retrieve": 0}
        self._instrument()

    def _instrument(self):
        a = self.assistant
        for kb in filter(None, [a.kb, a.price_agent.kb if a.price_agent else None]):
            def wrapped(query, min_score=None, _orig=kb.retrieve):
                found = _orig(query, min_score)
                self.count["retrieve"] += 1
                self.passages.extend(p.text for p in found)
                return found
            kb.retrieve = wrapped
        for name in ("converse", "converse_stream"):  # the price agent shares this client object
            def counted(*args, _orig=getattr(a.client, name), **kw):
                self.count["llm"] += 1
                return _orig(*args, **kw)
            setattr(a.client, name, counted)

    def ask(self, question, history=()):
        self.passages.clear()
        self.count.update(llm=0, retrieve=0)
        session = chat.Session(Path("eval/.unused"))  # never saved
        for turn in history:
            session.add("user", turn["user"], "eval")
            session.add("assistant", turn["assistant"], "eval")
        session.add("user", question, "eval")
        started = time.perf_counter()
        visible = join_visible(self.assistant.stream_reply(session))
        return Result(text=visible,
                      agents={s["agent"] for s in self.assistant.last_sources},
                      passages=list(self.passages), llm_calls=self.count["llm"], retrievals=self.count["retrieve"],
                      latency_s=time.perf_counter() - started)


TARGETS = {"assistant": AssistantTarget}


def join_visible(events):
    """Everything the user sees from (kind, agent, payload) events. Text events are streamed token chunks and must be
    joined with NOTHING between them ("7" + ".5%" is 7.5%); a sub-agent block is set apart on its own lines."""
    parts = [payload if kind == "text" else f"\n{payload}\n" for kind, _agent, payload in events
             if kind in ("text", "sub")]
    return "".join(parts).strip()


# ---- checks --------------------------------------------------------------------------------------------------------

def route_of(agents):
    routes = {ROUTE_OF_AGENT[a] for a in agents if a in ROUTE_OF_AGENT}
    return "both" if routes == {"exposure", "price"} else (routes.pop() if routes else "none")


def _matches(pattern, text):
    if pattern.startswith("re:"):
        return re.search(pattern[3:], text, re.I) is not None
    return pattern.lower() in text.lower()


def _figures(text, strict):
    """Normalised numeric values in text. strict=True keeps only money, percentages and k/M figures (what an answer
    can invent); strict=False keeps every number (what the sources and the question can legitimately contain)."""
    values = set()
    for m in _NUMBER.finditer(text):
        number, million, km, pct = m.groups()
        value = float(number.replace(",", ""))
        value *= 1e6 if million or (km and km.lower() == "m") else 1e3 if km else 1
        if strict:
            before = text[max(0, m.start() - 1):m.start()]
            if not (before == "$" or pct or km or million):
                continue
        values.add(round(value, 4))
    return values


def check_case(case, result):
    """Returns {route, answer, grounded: bool, problems: [str]}."""
    problems = []
    observed = route_of(result.agents)
    route_ok = observed == case["route"]
    if not route_ok:
        problems.append(f"route expected {case['route']}, got {observed}")

    for item in case.get("must_include", []):
        options = item if isinstance(item, list) else [item]
        if not any(_matches(o, result.text) for o in options):
            problems.append(f"missing {' | '.join(options)!r}")
    for item in case.get("must_not_include", []):
        if _matches(item, result.text):
            problems.append(f"must not contain {item!r}")
    if case.get("refuse") and not REFUSAL.search(result.text):
        problems.append("expected a refusal / not-available statement")
    answer_ok = not any(p.startswith(("missing", "must not", "expected a refusal")) for p in problems)

    allowed = _figures(" ".join(result.passages), strict=False)
    allowed |= _figures(case["question"] + " " + " ".join(h["user"] + " " + h["assistant"] for h in case.get("history", [])),
                        strict=False)
    ungrounded = sorted(v for v in _figures(result.text, strict=True) if v not in allowed)
    if ungrounded:
        problems.append(f"figures not in any source: {ungrounded}")
    return {"route": route_ok, "answer": answer_ok, "grounded": not ungrounded, "observed_route": observed,
            "problems": problems}


# ---- running and reporting -----------------------------------------------------------------------------------------

def run(config_path, profile, cases_path, label="run", repeat=1, name_filter=None, target="assistant"):
    cfg = chat.load_config(config_path)
    cases = yaml.safe_load(Path(cases_path).read_text(encoding="utf-8"))
    if name_filter:
        cases = [c for c in cases if name_filter.lower() in (c["id"] + " " + c["category"]).lower()]
    impl = TARGETS[target](cfg, profile)

    print(f"Evaluating target '{target}' on {len(cases)} cases x {repeat} run(s)\n")
    records = []
    for case in cases:
        runs = []
        for _ in range(repeat):
            result = impl.ask(case["question"], case.get("history", []))
            checks = check_case(case, result)
            runs.append({**checks, "passed": checks["route"] and checks["answer"] and checks["grounded"],
                         "latency_s": round(result.latency_s, 2), "llm_calls": result.llm_calls,
                         "retrievals": result.retrievals, "text": result.text})
        rate = sum(r["passed"] for r in runs) / repeat
        records.append({"id": case["id"], "category": case["category"], "question": case["question"],
                        "expected_route": case["route"], "pass_rate": rate, "runs": runs})
        worst = next((r for r in runs if not r["passed"]), runs[0])
        status = "PASS" if rate == 1 else ("FLAKY" if rate > 0 else "FAIL")
        print(f"{status:5} {case['id']:<14} {case['route']:>8} -> {worst['observed_route']:<8} "
              f"{sum(r['latency_s'] for r in runs) / repeat:5.1f}s {sum(r['llm_calls'] for r in runs) / repeat:3.1f} llm"
              + (f"  pass {rate:.0%}" if repeat > 1 else ""))
        for problem in worst["problems"] if rate < 1 else []:
            print(f"        - {problem}")

    report = {"label": label, "target": target, "when": datetime.now().isoformat(timespec="seconds"),
              "repeat": repeat, "cases": records}
    out = Path(cases_path).parent / "results"
    out.mkdir(exist_ok=True)
    path = out / f"{datetime.now():%Y%m%d-%H%M%S}-{label}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(report)
    print(f"\nSaved {path}")
    return report


def summarise(report):
    cases = report["cases"]
    runs = [r for c in cases for r in c["runs"]]
    n = len(runs) or 1
    by_cat = {}
    for c in cases:
        by_cat.setdefault(c["category"], []).append(c["pass_rate"])
    return {
        "pass": sum(c["pass_rate"] for c in cases) / (len(cases) or 1),
        "route": sum(r["route"] for r in runs) / n,
        "answer": sum(r["answer"] for r in runs) / n,
        "grounded": sum(r["grounded"] for r in runs) / n,
        "latency_s": sum(r["latency_s"] for r in runs) / n,
        "llm_calls": sum(r["llm_calls"] for r in runs) / n,
        "retrievals": sum(r["retrievals"] for r in runs) / n,
        "by_category": {k: sum(v) / len(v) for k, v in by_cat.items()},
    }


def print_summary(report):
    s = summarise(report)
    print(f"\n{'=' * 60}\nSCORE  {report['label']}  (target {report['target']}, {len(report['cases'])} cases x {report['repeat']})")
    print(f"  overall pass     {s['pass']:.0%}")
    print(f"  route accuracy   {s['route']:.0%}   answer accuracy {s['answer']:.0%}   grounded {s['grounded']:.0%}")
    print(f"  avg latency      {s['latency_s']:.1f}s   avg LLM calls {s['llm_calls']:.1f}   avg retrievals {s['retrievals']:.1f}")
    print("  by category      " + "   ".join(f"{k} {v:.0%}" for k, v in s["by_category"].items()))


def compare(path_a, path_b):
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (path_a, path_b))
    sa, sb = summarise(a), summarise(b)
    print(f"A = {a['label']} ({a['when']})\nB = {b['label']} ({b['when']})\n")
    print(f"{'metric':<18}{'A':>9}{'B':>9}")
    for key, fmt in (("pass", "{:.0%}"), ("route", "{:.0%}"), ("answer", "{:.0%}"), ("grounded", "{:.0%}"),
                     ("latency_s", "{:.1f}"), ("llm_calls", "{:.1f}"), ("retrievals", "{:.1f}")):
        print(f"{key:<18}{fmt.format(sa[key]):>9}{fmt.format(sb[key]):>9}")
    for cat in sorted(set(sa["by_category"]) | set(sb["by_category"])):
        print(f"{'  ' + cat:<18}{sa['by_category'].get(cat, float('nan')):>9.0%}{sb['by_category'].get(cat, float('nan')):>9.0%}")
    rates_a = {c["id"]: c["pass_rate"] for c in a["cases"]}
    changes = [(c["id"], rates_a[c["id"]], c["pass_rate"]) for c in b["cases"]
               if c["id"] in rates_a and rates_a[c["id"]] != c["pass_rate"]]
    print("\nCase changes (A -> B):")
    for cid, ra, rb in changes:
        print(f"  {'FIXED    ' if rb > ra else 'REGRESSED'} {cid}: {ra:.0%} -> {rb:.0%}")
    if not changes:
        print("  none")
