"""Choose min_score for each knowledge base from labelled questions.

# improvement 2 ranking step
For every question the KB is queried (retrieval + rerank) and the best rerank score is recorded. A good min_score sits
between the highest score of an out-of-scope question and the lowest score of an in-scope one. This prints the numbers
and a suggestion; it never edits config.yaml. Run:  python main.py --calibrate [eval/relevance_set.yaml]
"""
import math
from pathlib import Path

import yaml

from . import chat
from .assistant import Assistant


def _top_score(kb, question):
    found = kb.retrieve(question, min_score=0)  # min_score=0: see everything the reranker returns
    return found[0].score if found else 0.0


def run(config_path, profile, eval_path):
    cfg = chat.load_config(config_path)
    assistant = Assistant(cfg, cfg["chat"]["default_model"], profile)
    data = yaml.safe_load(Path(eval_path).read_text(encoding="utf-8"))
    if not assistant.reranker:
        print("WARNING: rerank.enabled is false: these are raw vector scores, which do not separate relevant from "
              "irrelevant text.\n")

    kbs = {"exposure": assistant.kb, "price": assistant.price_agent.kb if assistant.price_agent else None}
    for name, kb in kbs.items():
        if kb is None or name not in data:
            print(f"[{name}] skipped (disabled or no questions in {eval_path})\n")
            continue
        rows = {label: sorted(((_top_score(kb, q), q) for q in data[name].get(label, [])), reverse=True)
                for label in ("in_scope", "out_of_scope")}
        print(f"=== {name} KB {kb.id} | current min_score = {kb.min_score} ===")
        for label, items in rows.items():
            print(f"  {label}:")
            for score, q in items:
                verdict = "" if (score >= kb.min_score) == (label == "in_scope") else "   <-- WRONG at current min_score"
                print(f"    {score:6.3f}  {q}{verdict}")

        if not rows["in_scope"] or not rows["out_of_scope"]:
            print("  need both in_scope and out_of_scope questions to suggest a threshold\n")
            continue
        lowest_in, highest_out = rows["in_scope"][-1][0], rows["out_of_scope"][0][0]
        if lowest_in > highest_out:
            # geometric midpoint: rerank scores are skewed toward 0, so the arithmetic mean sits too close to the top
            suggestion = math.sqrt(lowest_in * highest_out) if highest_out > 0 else lowest_in / 2
            print(f"  separable: lowest in-scope {lowest_in:.3f} > highest out-of-scope {highest_out:.3f}")
            print(f"  SUGGESTED min_score = {suggestion:.2f}\n")
        else:
            print(f"  OVERLAP: lowest in-scope {lowest_in:.3f} <= highest out-of-scope {highest_out:.3f}; no threshold "
                  "separates them.")
            print("  Review the questions and the KB content (is the in-scope answer really in a chunk?).\n")
