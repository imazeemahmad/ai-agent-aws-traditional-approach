"""Shared fakes for the offline tests: no AWS calls are ever made."""
import contextlib
import copy
import io
import os
import tempfile
from pathlib import Path
from unittest import mock

from src import chat
from src.assistant import Assistant
from src.kb import Passage

SESSION_DIR = Path(tempfile.gettempdir()) / "claude_ai_tests"  # sessions are built in memory and never saved


def make_assistant():
    """The real Assistant wired from config.yaml, with dummy KB ids. Its AWS clients exist but are replaced by fakes."""
    cfg = copy.deepcopy(chat.load_config())
    cfg["aws"]["profile"] = None
    cfg["knowledge_base"]["knowledge_base_id"] = "EXPOSURE01"
    cfg["price_agent"]["knowledge_base_id"] = "PRICEKB001"
    env = {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test"}
    with mock.patch.dict(os.environ, env):
        os.environ.pop("BEDROCK_KB_ID", None)
        os.environ.pop("BEDROCK_PRICE_KB_ID", None)
        return Assistant(cfg, "haiku")


class FakeKB:
    """Stands in for KnowledgeBase: returns fixed passages and records the queries it received."""
    min_score = 0.4

    def __init__(self, rows=()):
        self.rows, self.queries = list(rows), []

    def retrieve(self, query, min_score=None):
        self.queries.append(query)
        return [Passage(text, name, "s3://bucket/" + name, score) for text, name, score in self.rows]


class FakeLLM:
    """Stands in for the bedrock-runtime client. `streams` are consumed one per converse_stream call."""

    def __init__(self, streams=(), converse_text="Whiteley oil, sold AUD 1,200,000 hammer [P1]."):
        self.streams, self.converse_text = list(streams), converse_text
        self.stream_calls, self.converse_calls = [], []

    def converse(self, **kwargs):
        self.converse_calls.append(kwargs)
        return {"output": {"message": {"content": [{"text": self.converse_text}]}}}

    def converse_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return {"stream": self.streams.pop(0)}


def text_stream(*chunks):
    """A model reply streamed as the given text chunks."""
    events = [{"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": c}}} for c in chunks]
    return events + [{"messageStop": {"stopReason": "end_turn"}}]


def tool_stream(question="Whiteley oil prices 2024", preamble="Checking prices.", tool_id="t1"):
    """A model reply that says `preamble`, then calls ask_price_agent(question)."""
    return [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": preamble}}},
        {"contentBlockStart": {"contentBlockIndex": 1,
                               "start": {"toolUse": {"toolUseId": tool_id, "name": "ask_price_agent"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1,
                               "delta": {"toolUse": {"input": '{"question": "' + question[:10]}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": question[10:] + '"}'}}}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]


def session_with(question, history=()):
    """An in-memory chat.Session holding earlier (user, assistant) turns and then the new question."""
    session = chat.Session(SESSION_DIR)
    for user, assistant in history:
        session.add("user", user, "haiku")
        session.add("assistant", assistant, "haiku")
    session.add("user", question, "haiku")
    return session


def render(assistant, session):
    """Run one question through the assistant and capture what the CLI would print."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        text, final_agent = chat.render_reply(assistant, assistant.stream_reply(session))
    return buffer.getvalue(), text, final_agent
