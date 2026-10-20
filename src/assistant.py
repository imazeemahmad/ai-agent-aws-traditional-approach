"""Orchestration: the exposure assistant (supervisor) and the price sub-agent, each grounded on its own Knowledge Base.

Flow for one question (grounded mode):
  1. Rewrite follow-ups into a standalone query.
  2. Retrieve from the exposure KB.
     - nothing relevant -> try the price sub-agent directly; nothing there either -> refuse (no supervisor LLM call).
     - relevant        -> supervisor LLM answers from the passages and may call the `ask_price_agent` tool.
  3. The price sub-agent retrieves from the price KB and answers only from those passages.
"""
import json
from pathlib import Path

import boto3

from .kb import KnowledgeBase, format_sources
from .rerank import Reranker  # improvement 2 ranking step

BASE = Path(__file__).resolve().parent.parent  # project root (config.yaml, certs/, sessions/ live here)

PRICE_TOOL = {
    "tools": [{
        "toolSpec": {
            "name": "ask_price_agent",
            "description": "Ask the artwork price specialist, which answers only from a separate price knowledge base "
                           "(auction results, sale prices, estimates). Use it for any question about prices, sale "
                           "results or values of artists or artworks.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"question": {
                    "type": "string",
                    "description": "A self-contained price question naming the artist, work, medium and period, "
                                   "e.g. 'Hammer prices for Brett Whiteley oil paintings sold in Sydney 2023-2025'."}},
                "required": ["question"],
            }},
        }
    }]
}


def _text(resp):
    """Concatenate the text blocks of a non-streaming Converse response."""
    return "".join(b.get("text", "") for b in resp["output"]["message"]["content"]).strip()


class PriceAgent:
    """Stateless specialist: one standalone question in, one grounded answer (with [P#] citations) out."""

    def __init__(self, boto_session, cfg, client, verify, model_for, reranker=None):
        self.cfg = cfg
        # improvement 2 ranking step: the price KB shares the reranker, so its min_score is a rerank score too
        self.kb = KnowledgeBase(boto_session, cfg, verify, env_var="BEDROCK_PRICE_KB_ID", label="price_agent",
                                reranker=reranker)
        self.client = client
        self.model_for = model_for  # model key (or None for the active model) -> model settings

    def answer(self, question, start=1):
        """Returns (text, passages). Empty passages means the price KB has nothing relevant (no LLM call is made)."""
        passages = self.kb.retrieve(question)
        if not passages:
            return self.cfg["no_data_message"].strip(), []
        m = self.model_for(self.cfg.get("model"))
        grounded = f"<sources>\n{format_sources(passages, 'P', start)}\n</sources>\n\nQuestion: {question}"
        resp = self.client.converse(
            modelId=m["model_id"],
            system=[{"text": self.cfg["system_prompt"]}],
            messages=[{"role": "user", "content": [{"text": grounded}]}],
            inferenceConfig={"maxTokens": m["max_tokens"], "temperature": 0},
        )
        return _text(resp), passages


class Assistant:
    def __init__(self, cfg, model_key, profile=None):
        self.cfg = cfg
        self.model_key = model_key
        boto_session = boto3.Session(profile_name=profile or cfg["aws"].get("profile"), region_name=cfg["aws"]["region"])
        ca_bundle = cfg["aws"].get("ca_bundle")  # corporate proxy CA (TLS inspection); None = default/AWS_CA_BUNDLE
        verify = str(BASE / ca_bundle) if ca_bundle else None
        self.client = boto_session.client("bedrock-runtime", verify=verify)
        # improvement 2 ranking step: one reranker shared by both knowledge bases (None = vector scores only)
        rerank_cfg = cfg.get("rerank") or {}
        self.reranker = Reranker(boto_session, rerank_cfg, verify) if rerank_cfg.get("enabled") else None
        kb_cfg = cfg.get("knowledge_base") or {}
        self.kb = KnowledgeBase(boto_session, kb_cfg, verify, reranker=self.reranker) if kb_cfg.get("enabled") else None
        pa_cfg = cfg.get("price_agent") or {}
        self.price_agent = (PriceAgent(boto_session, pa_cfg, self.client, verify, self._model_settings, self.reranker)
                            if self.kb and pa_cfg.get("enabled") else None)
        self.last_sources = []  # sources behind the most recent reply: [{id, agent, file, uri, score}]
        self.last_calls = []    # sub-agent calls made for the most recent reply: [{agent, question, answer, routed}]

    def _model_settings(self, key=None):
        return self.cfg["models"][key or self.model_key]

    def label(self, agent):
        """Display name of an agent with the model it runs on, e.g. 'price agent | haiku'."""
        if agent == "system":
            return "system"
        key = self.cfg["price_agent"].get("model") if agent == "price" else None
        return f"{agent} agent | {key or self.model_key}"

    def switch(self, key):
        if key not in self.cfg["models"]:
            raise KeyError(key)
        self.model_key = key

    def _record(self, passages, agent, prefix, start):
        for i, p in enumerate(passages, start):
            self.last_sources.append({"id": f"{prefix}{i}", "agent": agent, "file": p.source,
                                      "uri": p.uri, "score": round(p.score, 4)})

    def _search_query(self, session):
        """The text to search the knowledge bases with: the question, made standalone if it is a follow-up."""
        question = session.messages[-1]["content"][0]["text"]
        prior = session.context(self.cfg["chat"]["max_history_turns"])[:-1]
        if not (self.cfg["knowledge_base"].get("rewrite_followups") and prior):
            return question
        transcript = "\n".join(f"{m['role']}: {m['content'][0]['text'][:500]}" for m in prior[-6:])
        resp = self.client.converse(
            modelId=self._model_settings()["model_id"],
            system=[{"text": "Rewrite the user's latest question as one standalone search query, resolving pronouns "
                             "and references using the conversation. Output only the query, nothing else."}],
            messages=[{"role": "user", "content": [{"text": f"Conversation:\n{transcript}\n\nLatest question: {question}"}]}],
            inferenceConfig={"maxTokens": 100, "temperature": 0},
        )
        return _text(resp) or question

    def stream_reply(self, session):
        messages = session.context(self.cfg["chat"]["max_history_turns"])
        system = self.cfg["system_prompt"]
        tool_config = None
        self.last_sources, self.last_calls = [], []

        if self.kb:  # grounded mode: retrieve first, answer only from what comes back
            kb_cfg = self.cfg["knowledge_base"]
            query = self._search_query(session)
            passages = self.kb.retrieve(query)

            if not passages:
                # Not an exposure question; it may still be a pure price question for the sub-agent.
                if self.price_agent:
                    text, price_passages = self.price_agent.answer(query)
                    if price_passages:
                        self._record(price_passages, "price", "P", 1)
                        self.last_calls.append({"agent": "price", "question": query, "answer": text, "routed": True})
                        yield "route", "price", "no exposure KB match; question routed directly to the price agent"
                        yield "text", "price", text
                        return
                # nothing relevant anywhere: refuse without any LLM call
                yield "text", "system", kb_cfg["no_answer_message"].strip()
                return

            self._record(passages, "main", "", 1)
            question = messages[-1]["content"][0]["text"]
            # sources go only into this request; the saved history keeps the plain question
            grounded = f"<sources>\n{format_sources(passages)}\n</sources>\n\nQuestion: {question}"
            messages = messages[:-1] + [{"role": "user", "content": [{"text": grounded}]}]
            system = kb_cfg["system_prompt"]
            if self.price_agent:
                system += "\n" + self.cfg["price_agent"]["supervisor_prompt"]
                tool_config = PRICE_TOOL

        yield from self._converse_loop(system, messages, tool_config)  # events: (kind, agent, payload)

    def _converse_loop(self, system, messages, tool_config):
        """Stream the model's answer, executing tool calls (the price sub-agent) until it produces a final answer."""
        m = self._model_settings()
        max_rounds = self.cfg.get("price_agent", {}).get("max_tool_rounds", 3) if tool_config else 0

        for round_no in range(max_rounds + 1):
            kwargs = dict(
                modelId=m["model_id"],
                system=[{"text": system}],
                messages=messages,
                inferenceConfig={"maxTokens": m["max_tokens"], "temperature": m["temperature"]},
            )
            if tool_config:  # must stay set once toolUse/toolResult blocks are in the messages
                kwargs["toolConfig"] = tool_config
            blocks, stop = {}, None
            for event in self.client.converse_stream(**kwargs)["stream"]:
                if "contentBlockStart" in event:
                    tool = event["contentBlockStart"].get("start", {}).get("toolUse")
                    if tool:
                        blocks[event["contentBlockStart"]["contentBlockIndex"]] = {
                            "toolUse": {"toolUseId": tool["toolUseId"], "name": tool["name"]}, "json": ""}
                elif "contentBlockDelta" in event:
                    idx, delta = event["contentBlockDelta"]["contentBlockIndex"], event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        blocks.setdefault(idx, {"text": ""})["text"] += delta["text"]
                        yield "text", "main", delta["text"]
                    elif "toolUse" in delta:
                        blocks[idx]["json"] += delta["toolUse"].get("input", "")
                elif "messageStop" in event:
                    stop = event["messageStop"]["stopReason"]

            if stop != "tool_use":
                return

            content, calls = [], []
            for idx in sorted(blocks):
                b = blocks[idx]
                if "toolUse" in b:
                    call = {**b["toolUse"], "input": json.loads(b["json"] or "{}")}
                    content.append({"toolUse": call})
                    calls.append(call)
                elif b["text"]:
                    content.append({"text": b["text"]})
            results = []
            for c in calls:
                yield "call", "price", str(c["input"].get("question", "")).strip()
                answer = self._run_tool(c)
                yield "sub", "price", answer
                results.append({"toolResult": {"toolUseId": c["toolUseId"], "content": [{"text": answer}]}})
            messages = messages + [{"role": "assistant", "content": content}, {"role": "user", "content": results}]

        yield "text", "main", "[stopped: too many tool calls]"

    def _run_tool(self, call):
        if call["name"] != "ask_price_agent" or not self.price_agent:
            return f"ERROR: unknown tool {call['name']}"
        question = str(call["input"].get("question", "")).strip()
        if not question:
            return "ERROR: question is required"
        start = 1 + sum(1 for s in self.last_sources if s["agent"] == "price")  # keep [P#] ids unique within a reply
        text, passages = self.price_agent.answer(question, start)
        self._record(passages, "price", "P", start)
        self.last_calls.append({"agent": "price", "question": question, "answer": text, "routed": False})
        return text
