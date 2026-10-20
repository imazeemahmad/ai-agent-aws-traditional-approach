"""The evaluation scoring itself: if these checks are wrong, every score is wrong."""
import unittest
from pathlib import Path

import yaml

from src.evaluate import Result, _figures, check_case, join_visible, route_of

ROOT = Path(__file__).resolve().parent.parent
CASES = {c["id"]: c for c in yaml.safe_load((ROOT / "eval" / "cases.yaml").read_text(encoding="utf-8"))}


def check(case_id, text, agents, passages=()):
    return check_case(CASES[case_id], Result(text=text, agents=set(agents), passages=list(passages)))


class CasesFileTests(unittest.TestCase):
    def test_case_set_is_complete_and_well_formed(self):
        self.assertEqual({c["category"] for c in CASES.values()}, {"exposure", "price", "both", "out_of_scope", "followup"})
        self.assertGreaterEqual(len(CASES), 20)
        for case in CASES.values():
            self.assertIn(case["route"], ("exposure", "price", "both", "none"), case["id"])
            self.assertTrue(case.get("must_include") or case.get("refuse"), f"{case['id']} checks nothing")


class ScoringTests(unittest.TestCase):
    def test_route_is_derived_from_the_agents_that_supplied_sources(self):
        self.assertEqual([route_of(a) for a in ({"main"}, {"price"}, {"main", "price"}, set())],
                         ["exposure", "price", "both", "none"])

    def test_figures_strict_finds_what_an_answer_can_invent_and_loose_finds_what_sources_contain(self):
        self.assertEqual(_figures("fee $10,000 and 7.5% and $500k and $2.5M, 3 items, stage 4", strict=True),
                         {10000.0, 7.5, 500000.0, 2500000.0})
        self.assertLessEqual({500000.0, 2500000.0, 7.5}, _figures("up to 500k 20% 2.5M 5M 7.5%", strict=False))

    def test_a_correct_grounded_answer_passes_every_check(self):
        r = check("price-02", "Fa Inni Qareeb costs $90.00 [P1].", {"price"}, ["Fa Inni Qareeb $90.00 Add to cart"])
        self.assertEqual((r["route"], r["answer"], r["grounded"], r["problems"]), (True, True, True, []))

    def test_wrong_value_fails_the_answer_and_the_grounding_check(self):
        r = check("price-02", "It costs $95 [P1].", {"price"}, ["Fa Inni Qareeb $90.00"])
        self.assertFalse(r["answer"])
        self.assertFalse(r["grounded"])

    def test_right_answer_from_the_wrong_agent_fails_only_the_route(self):
        r = check("price-02", "$90", {"main"}, ["$90"])
        self.assertEqual((r["route"], r["answer"]), (False, True))

    def test_speculation_guard_catches_the_known_price_failure(self):
        r = check("price-01", "It is $280. The $490 one may represent a premium version.", {"price"}, ["$280 $490"])
        self.assertFalse(r["answer"])
        self.assertTrue(any("must not" in p for p in r["problems"]))

    def test_k_and_m_figures_are_grounded_against_table_text(self):
        r = check("both-02", "Up to $500k the curator fee is 20% [1]; Fazkurooni is $190 [P1].", {"main", "price"},
                  ["up to 500k 20%", "Fazkurooni $190.00"])
        self.assertEqual((r["route"], r["answer"], r["grounded"]), (True, True, True))

    def test_refusals(self):
        self.assertTrue(all(check("oos-01", "I couldn't find that in the knowledge base, so I can't answer it.", set())[k]
                            for k in ("route", "answer", "grounded")))
        self.assertFalse(check("oos-01", "The Cats won.", set())["answer"])                       # answered instead of refusing
        self.assertFalse(check("oos-05", "I couldn't find that. Leonardo da Vinci painted it.", set())["answer"])  # leaked
        r = check("oos-03", "He is well exposed. Sources [1].", {"main"})                          # invented from irrelevant sources
        self.assertEqual((r["route"], r["answer"]), (False, False))

    def test_numbers_from_the_conversation_history_count_as_grounded(self):
        r = check("followup-01", "Fazkurooni is $190 [P1], versus $90 for Fa Inni Qareeb.", {"price"}, ["Fazkurooni $190.00"])
        self.assertTrue(r["grounded"])

    def test_streamed_chunks_are_joined_without_separators(self):
        # regression: joining token chunks with a space turned "7.5%" into "7 .5%" and produced false failures
        text = join_visible([("text", "main", "The fee is "), ("text", "main", "7"), ("text", "main", ".5%"),
                             ("route", "price", "ignored"), ("call", "price", "q"), ("sub", "price", "$25,000 [P1]"),
                             ("text", "main", " Done.")])
        self.assertIn("7.5%", text)
        self.assertIn("$25,000 [P1]", text)
        self.assertNotIn("ignored", text)
        self.assertTrue(check("exposure-02", text, {"main"}, ["Over 5M 7.5%"])["answer"])


if __name__ == "__main__":
    unittest.main()
