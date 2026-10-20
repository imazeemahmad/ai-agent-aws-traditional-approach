"""Knowledge base retrieval, reranking and relevance gating (offline, fake bedrock-agent-runtime client)."""
import unittest

from botocore.exceptions import ClientError

from src.kb import KnowledgeBase
from src.rerank import Reranker


class FakeSession:
    region_name = "us-east-1"

    def __init__(self, client):
        self._client = client

    def client(self, *args, **kwargs):
        return self._client


class FakeAgentRuntime:
    """Retrieve returns near-identical vector scores (like the real KBs); Rerank rates only chunks 3 and 7 relevant."""

    def __init__(self, wants="vectorSearchConfiguration"):
        self.wants, self.retrieve_calls, self.rerank_calls = wants, [], []

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        key = next(iter(kwargs["retrievalConfiguration"]))
        if key != self.wants:
            raise ClientError({"Error": {"Code": "ValidationException",
                                         "Message": f"Incompatible configuration: {key} is not supported. Use {self.wants} instead."}},
                              "Retrieve")
        count = kwargs["retrievalConfiguration"][key]["numberOfResults"]
        return {"retrievalResults": [{"content": {"text": f"chunk {i}"}, "score": 0.9 - i * 0.001,
                                      "location": {"s3Location": {"uri": f"s3://b/doc{i}.pdf"}}} for i in range(count)]}

    def rerank(self, **kwargs):
        self.rerank_calls.append(kwargs)
        top = kwargs["rerankingConfiguration"]["bedrockRerankingConfiguration"]["numberOfResults"]
        scores = {3: 0.8, 7: 0.3}
        results = sorted(({"index": i, "relevanceScore": scores.get(i, 0.01)} for i in range(len(kwargs["sources"]))),
                         key=lambda r: -r["relevanceScore"])
        return {"results": results[:top]}


def make_kb(wants="vectorSearchConfiguration", rerank=True, **cfg):
    client = FakeAgentRuntime(wants)
    session = FakeSession(client)
    settings = {"knowledge_base_id": "ABCDE12345", "num_results": 5, "fetch_k": 20, "min_score": 0.2, **cfg}
    reranker = Reranker(session, {"model_id": "cohere.rerank-v3-5:0"}) if rerank else None
    return KnowledgeBase(session, settings, reranker=reranker), client


class RetrievalTests(unittest.TestCase):
    def test_fetches_fetch_k_reranks_and_filters_on_the_rerank_score(self):
        kb, client = make_kb()
        found = kb.retrieve("anything")
        self.assertEqual(client.retrieve_calls[0]["retrievalConfiguration"]["vectorSearchConfiguration"]["numberOfResults"], 20)
        self.assertEqual([p.text for p in found], ["chunk 3", "chunk 7"])
        self.assertEqual(found[0].score, 0.8)                      # relevance is the rerank score...
        self.assertAlmostEqual(found[0].vector_score, 0.897)       # ...the original vector score is kept for debugging
        arn = client.rerank_calls[0]["rerankingConfiguration"]["bedrockRerankingConfiguration"]["modelConfiguration"]["modelArn"]
        self.assertEqual(arn, "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0")

    def test_irrelevant_query_returns_nothing(self):
        kb, _ = make_kb(min_score=0.9)
        self.assertEqual(kb.retrieve("lasagne"), [])

    def test_min_score_zero_shows_the_top_results_for_tuning(self):
        kb, _ = make_kb()
        self.assertEqual(len(kb.retrieve("x", min_score=0)), 5)

    def test_managed_kb_is_detected_once_and_remembered(self):
        kb, client = make_kb(wants="managedSearchConfiguration")
        self.assertTrue(kb.retrieve("x"))
        self.assertEqual(len(client.retrieve_calls), 2)            # one rejected call, then the right setting
        self.assertEqual(kb.search_key, "managedSearchConfiguration")
        kb.retrieve("y")
        self.assertEqual(len(client.retrieve_calls), 3)            # no repeated failure

    def test_without_reranking_behaviour_is_the_plain_vector_search(self):
        kb, client = make_kb(rerank=False, min_score=0.0)
        found = kb.retrieve("x")
        self.assertEqual(client.retrieve_calls[0]["retrievalConfiguration"]["vectorSearchConfiguration"]["numberOfResults"], 5)
        self.assertEqual((len(found), client.rerank_calls), (5, []))

    def test_unrelated_validation_errors_are_not_swallowed(self):
        class Broken(FakeAgentRuntime):
            def retrieve(self, **kwargs):
                raise ClientError({"Error": {"Code": "ValidationException", "Message": "knowledgeBaseId invalid"}}, "Retrieve")

        kb = KnowledgeBase(FakeSession(Broken()), {"knowledge_base_id": "ABCDE12345"})
        with self.assertRaises(ClientError):
            kb.retrieve("x")

    def test_placeholder_knowledge_base_id_is_rejected_with_a_clear_message(self):
        with self.assertRaisesRegex(ValueError, "knowledge_base_id is not set"):
            KnowledgeBase(FakeSession(FakeAgentRuntime()), {"knowledge_base_id": "REPLACE_WITH_KB_ID"})


if __name__ == "__main__":
    unittest.main()
