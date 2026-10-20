"""How a question flows through the main agent and the price agent (offline, fake KBs and fake model)."""
import unittest

from tests.helpers import FakeKB, FakeLLM, make_assistant, render, session_with, text_stream, tool_stream

EXPOSURE = [("Gallery X represents him.", "galleries.pdf", 0.7)]
PRICE = [("AUD 1.2m", "prices-2024.csv", 0.8)]


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.a = make_assistant()

    def ask(self, question, exposure=(), price=(), streams=(), history=(), **llm_kwargs):
        """Wire fakes into the assistant, run one question, return (printed output, saved text, final agent, llm)."""
        self.a.kb, self.a.price_agent.kb = FakeKB(exposure), FakeKB(price)
        self.llm = FakeLLM(streams, **llm_kwargs)
        self.a.client = self.a.price_agent.client = self.llm
        out, text, final = render(self.a, session_with(question, history))
        return out, text, final

    def test_nothing_relevant_refuses_without_calling_the_llm(self):
        out, _, final = self.ask("Who won the AFL?")
        self.assertTrue(out.startswith("[system] "))
        self.assertEqual(final, "system")
        self.assertEqual((self.llm.stream_calls, self.llm.converse_calls), ([], []))
        self.assertEqual(self.a.last_sources, [])

    def test_pure_price_question_goes_straight_to_the_price_agent(self):
        out, text, final = self.ask("What did Whiteley sell for?", price=PRICE)
        self.assertIn("no exposure KB match; question routed directly to the price agent", out)
        self.assertIn("[price agent | haiku] Whiteley", out)
        self.assertEqual(final, "price")
        self.assertEqual(self.llm.stream_calls, [])               # the main agent's model is never called
        self.assertEqual(len(self.llm.converse_calls), 1)         # only the price agent
        self.assertEqual([s["id"] for s in self.a.last_sources], ["P1"])
        self.assertTrue(self.a.last_calls[0]["routed"])

    def test_exposure_question_is_answered_by_the_main_agent_with_the_tool_on_offer(self):
        out, text, final = self.ask("How exposed is he?", exposure=EXPOSURE, streams=[text_stream("Well ", "exposed [1].")])
        self.assertEqual(out.strip(), "[main agent | haiku] Well exposed [1].")
        self.assertEqual(final, "main")
        call = self.llm.stream_calls[0]
        self.assertIn("toolConfig", call)
        self.assertIn("ask_price_agent", call["system"][0]["text"])
        self.assertEqual([s["id"] for s in self.a.last_sources], ["1"])

    def test_retrieved_passages_go_in_the_request_but_not_in_the_saved_history(self):
        session = session_with("How exposed is he?")
        self.a.kb, self.a.price_agent.kb = FakeKB(EXPOSURE), FakeKB()
        self.llm = FakeLLM([text_stream("ok [1].")])
        self.a.client = self.a.price_agent.client = self.llm
        render(self.a, session)
        sent = self.llm.stream_calls[0]["messages"][-1]["content"][0]["text"]
        self.assertIn('<source id="1" file="galleries.pdf">', sent)
        self.assertEqual(session.messages[-1]["content"][0]["text"], "How exposed is he?")

    def test_main_agent_calls_the_price_agent_through_the_tool_and_both_are_labelled(self):
        out, text, final = self.ask(
            "Exposure and prices?", exposure=EXPOSURE, price=PRICE,
            streams=[tool_stream(), text_stream("Exposure high [1]. Prices AUD 1.2m [P1].")])
        self.assertEqual([line for line in out.split("\n") if line], [
            "[main agent | haiku] Checking prices.",
            "  -> main agent asks the price agent: 'Whiteley oil prices 2024'",
            "[price agent | haiku] Whiteley oil, sold AUD 1,200,000 hammer [P1].",
            "[main agent | haiku] Exposure high [1]. Prices AUD 1.2m [P1].",
        ])
        # the price agent's raw answer is displayed and audited, but only the main agent's text is saved to history
        self.assertEqual(text, "Checking prices.\n\nExposure high [1]. Prices AUD 1.2m [P1].")
        self.assertEqual(final, "main")
        self.assertEqual([s["id"] for s in self.a.last_sources], ["1", "P1"])
        self.assertEqual(self.a.last_calls, [{"agent": "price", "question": "Whiteley oil prices 2024",
                                              "answer": "Whiteley oil, sold AUD 1,200,000 hammer [P1].", "routed": False}])
        # the second model call carries the tool use and its result, and keeps the tool config (the API requires it)
        second = self.llm.stream_calls[1]
        self.assertIn("toolConfig", second)
        self.assertEqual(second["messages"][-2]["content"][-1]["toolUse"]["input"], {"question": "Whiteley oil prices 2024"})
        self.assertEqual(second["messages"][-1]["content"][0]["toolResult"]["toolUseId"], "t1")
        self.assertEqual(self.a.price_agent.kb.queries, ["Whiteley oil prices 2024"])

    def test_price_citation_ids_stay_unique_across_several_tool_calls(self):
        self.ask("Two prices?", exposure=EXPOSURE, price=PRICE + [("AUD 2m", "prices-2023.csv", 0.7)],
                 streams=[tool_stream(tool_id="t1"), tool_stream(tool_id="t2"), text_stream("done")])
        self.assertEqual([s["id"] for s in self.a.last_sources], ["1", "P1", "P2", "P3", "P4"])

    def test_runaway_tool_calls_are_stopped(self):
        rounds = self.a.cfg["price_agent"]["max_tool_rounds"]
        out, _, _ = self.ask("Loop?", exposure=EXPOSURE, price=PRICE,
                             streams=[tool_stream(tool_id=f"t{i}") for i in range(rounds + 1)])
        self.assertIn("[stopped: too many tool calls]", out)
        self.assertEqual(len(self.llm.stream_calls), rounds + 1)

    def test_without_the_price_agent_there_is_no_tool_and_no_direct_route(self):
        self.a.price_agent = None
        self.a.kb = FakeKB(EXPOSURE)
        self.a.client = self.llm = FakeLLM([text_stream("ok [1].")])
        render(self.a, session_with("How exposed is he?"))
        self.assertNotIn("toolConfig", self.llm.stream_calls[0])
        self.a.kb = FakeKB()
        out, _, final = render(self.a, session_with("What did it sell for?"))
        self.assertEqual(final, "system")

    def test_follow_up_is_rewritten_before_retrieval_but_a_first_question_is_not(self):
        self.ask("How exposed is he?", exposure=EXPOSURE, streams=[text_stream("ok")])
        self.assertEqual(self.llm.converse_calls, [])                         # first turn: no rewrite call
        self.ask("and its sales?", exposure=EXPOSURE, streams=[text_stream("ok")],
                 history=[("Tell me about Sydney Contemporary", "It is an art fair [1].")],
                 converse_text="Sydney Contemporary art fair sales")
        self.assertEqual(self.a.kb.queries, ["Sydney Contemporary art fair sales"])


if __name__ == "__main__":
    unittest.main()
