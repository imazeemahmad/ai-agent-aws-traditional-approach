"""Bedrock Knowledge Base retrieval: the only source of facts the assistant may use."""
import os
from dataclasses import dataclass
from pathlib import PurePosixPath

from botocore.exceptions import ClientError

# improvement 2 ranking step: retrieval-config key per knowledge base type
SEARCH_KEYS = {"vector": "vectorSearchConfiguration", "managed": "managedSearchConfiguration"}

@dataclass
class Passage:
    text: str
    source: str   # file name, e.g. "auction-report-2025.pdf"
    uri: str      # full location, e.g. s3://bucket/path/auction-report-2025.pdf
    score: float  # relevance used for filtering and display: the rerank score when reranking is on
    vector_score: float = 0.0  # improvement 2 ranking step: original vector similarity, kept for debugging (/kb)


class KnowledgeBase:
    def __init__(self, boto_session, cfg, verify=None, env_var="BEDROCK_KB_ID", label="knowledge_base", reranker=None):
        self.id = os.environ.get(env_var) or cfg.get("knowledge_base_id")
        if not self.id or str(self.id).startswith("REPLACE"):
            raise ValueError(f"{label}.knowledge_base_id is not set in config.yaml (or {env_var} env var)")
        self.num_results = cfg.get("num_results", 5)
        self.min_score = cfg.get("min_score", 0.0)  # improvement 2 ranking step: compared with the RERANK score when on
        self.reranker = reranker
        # improvement 2 ranking step: fetch a wider candidate set, let the reranker pick the best num_results
        self.fetch_k = max(cfg.get("fetch_k", 20), self.num_results) if reranker else self.num_results
        # bedrock-agent-runtime is a different endpoint from bedrock-runtime: it needs the same proxy CA bundle
        self.client = boto_session.client("bedrock-agent-runtime", verify=verify)
        # improvement 2 ranking step: standard vector KBs and managed KBs take different retrieval settings and each
        # rejects the other's. Start with the config value (vector | managed); retrieve() self-corrects and remembers.
        self.search_key = SEARCH_KEYS.get(cfg.get("search_type", "vector"), SEARCH_KEYS["vector"])

    def _retrieve_raw(self, query):
        """Call Retrieve for fetch_k candidates, switching the search-config key once if the KB type needs the other."""
        for attempt in (1, 2):
            try:
                return self.client.retrieve(
                    knowledgeBaseId=self.id,
                    retrievalQuery={"text": query[:1000]},  # Retrieve limits the query text length
                    retrievalConfiguration={self.search_key: {"numberOfResults": self.fetch_k}},  # improvement 2 ranking step: fetch_k, not num_results
                )
            except ClientError as e:
                err = e.response["Error"]
                other = next(k for k in SEARCH_KEYS.values() if k != self.search_key)
                if attempt == 1 and err["Code"] == "ValidationException" and other in err["Message"]:
                    self.search_key = other  # improvement 2 ranking step: the KB told us which one it wants
                    continue
                raise

    def retrieve(self, query, min_score=None):
        """Semantic search, then rerank. Returns passages scoring >= min_score, best first."""
        resp = self._retrieve_raw(query)
        candidates = []
        for r in resp.get("retrievalResults", []):
            text = r.get("content", {}).get("text", "").strip()
            if not text:
                continue
            uri = r.get("location", {}).get("s3Location", {}).get("uri", "") or r.get("location", {}).get("type", "unknown")
            candidates.append(Passage(text, PurePosixPath(uri).name or uri, uri, r.get("score", 0.0)))

        if self.reranker:  # improvement 2 ranking step: replace vector scores with calibrated rerank scores
            candidates = self.reranker.rerank(query, candidates, self.num_results)

        threshold = self.min_score if min_score is None else min_score  # improvement 2 ranking step: applied after reranking
        passages = [p for p in candidates if p.score >= threshold]
        return sorted(passages, key=lambda p: p.score, reverse=True)[:self.num_results]


def format_sources(passages, prefix="", start=1):
    """<source> blocks the model must cite by id: [1], [2]... (or [P1], [P2]... with prefix="P")."""
    return "\n".join(f'<source id="{prefix}{i}" file="{p.source}">\n{p.text}\n</source>'
                     for i, p in enumerate(passages, start))
