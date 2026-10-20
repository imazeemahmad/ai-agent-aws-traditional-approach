"""Reranking of retrieved passages with the Bedrock Rerank API.

# improvement 2 ranking step
Why: vector-similarity scores from the knowledge bases do not separate relevant from irrelevant text (a lasagne recipe
scored 0.92 against the art KBs, the same as real matches), so min_score could never filter anything. A reranking model
reads the query and each passage together and returns a calibrated relevance score (about 0.6 for a real match,
below 0.01 for unrelated text), which makes min_score, the refusal gate and the direct price routing meaningful.
"""


class Reranker:
    def __init__(self, boto_session, cfg, verify=None):
        # improvement 2 ranking step: same region as the knowledge bases, so passages never leave the region
        self.model_arn = cfg.get("model_arn") or f"arn:aws:bedrock:{boto_session.region_name}::foundation-model/{cfg['model_id']}"
        self.client = boto_session.client("bedrock-agent-runtime", verify=verify)

    def rerank(self, query, passages, top_n):
        """Score passages against the query. Returns the best top_n, best first.

        Each returned passage has score = rerank relevance, vector_score = the original vector similarity.
        """
        if not passages:  # the API rejects an empty source list
            return []
        resp = self.client.rerank(  # improvement 2 ranking step
            queries=[{"type": "TEXT", "textQuery": {"text": query[:1000]}}],
            sources=[{"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": p.text}}}
                     for p in passages],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {"modelArn": self.model_arn},
                    "numberOfResults": min(top_n, len(passages)),
                },
            },
        )
        ranked = []
        for r in resp["results"]:
            p = passages[r["index"]]
            p.vector_score, p.score = p.score, r["relevanceScore"]  # improvement 2 ranking step
            ranked.append(p)
        return sorted(ranked, key=lambda p: p.score, reverse=True)
