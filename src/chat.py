"""Sydney artwork market exposure assistant - CLI POC on AWS Bedrock (Converse API)."""
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from botocore.exceptions import BotoCoreError, ClientError

from .assistant import BASE, Assistant

CONFIG_PATH = BASE / "config.yaml"


HELP = """Commands:
  /models          list configured models
  /model <name>    switch model (context is kept)
  /kb <question>        test exposure KB retrieval (passages and scores)
  /kb price <question>  test price KB retrieval
  /history         show this session's messages
  /new             start a new session
  /sessions        list saved sessions
  /help            show this help
  /quit            exit"""


def load_config(path=CONFIG_PATH):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class Session:
    """One conversation: id, message history and the model used for each reply."""

    def __init__(self, session_dir, session_id=None):
        self.dir = session_dir
        self.id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.created = datetime.now(timezone.utc).isoformat()
        self.messages = []  # Converse-format messages
        self.log = []       # audit trail: timestamp, model, role, text

    @property
    def path(self):
        return self.dir / f"{self.id}.json"

    def add(self, role, text, model, sources=None, **meta):
        self.messages.append({"role": role, "content": [{"text": text}]})
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "role": role, "model": model, "text": text}
        if sources:
            entry["sources"] = sources  # knowledge base passages this answer was grounded on (audit trail)
        entry.update({k: v for k, v in meta.items() if v})  # e.g. agent, calls (sub-agent questions and answers)
        self.log.append(entry)

    def pop_last(self):
        self.messages.pop()
        self.log.pop()

    def save(self):
        self.dir.mkdir(exist_ok=True)
        data = {"id": self.id, "created": self.created, "messages": self.messages, "log": self.log}
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, session_dir, session_id):
        data = json.loads((session_dir / f"{session_id}.json").read_text(encoding="utf-8"))
        s = cls(session_dir, data["id"])
        s.created, s.messages, s.log = data["created"], data["messages"], data["log"]
        return s

    def context(self, max_turns):
        """Last N turns, always starting on a user message (Converse requirement)."""
        msgs = self.messages[-max_turns * 2:]
        while msgs and msgs[0]["role"] != "user":
            msgs = msgs[1:]
        return msgs


def render_reply(assistant, events):
    """Print the reply, labelling which agent produced each part. Returns (text to save, agent of the final answer).

    Events are (kind, agent, payload): text = answer chunk; route / call = hand-off notes; sub = a sub-agent's answer.
    Only "text" is saved to the history; sub-agent answers are shown and kept in the audit log (calls).
    """
    saved, current, line_open, final = [], None, False, "main"
    for kind, agent, payload in events:
        if kind == "text":
            if current != agent:
                print(("\n" if line_open else "") + f"[{assistant.label(agent)}] ", end="")
                current = agent
                if saved:
                    saved.append("\n\n")  # keep segments apart in the saved history
            print(payload, end="", flush=True)
            line_open = True
            saved.append(payload)
            final = agent
            continue
        if line_open:
            print()
            line_open = False
        if kind == "route":
            print(f"  -> {payload}")
        elif kind == "call":
            print(f"  -> main agent asks the price agent: {payload!r}")
        elif kind == "sub":
            print(f"[{assistant.label(agent)}] {payload}")
        current = None
    if line_open:
        print()
    print()
    return "".join(saved), final


def print_sessions(session_dir):
    files = sorted(session_dir.glob("*.json")) if session_dir.exists() else []
    if not files:
        print("No saved sessions.")
    for f in files:
        print(f"  {f.stem}")


def run(config_path=CONFIG_PATH, profile=None, model=None, session_id=None):
    """Start the interactive chat loop."""
    cfg = load_config(config_path)
    session_dir = BASE / cfg["chat"]["session_dir"]
    model_key = model or cfg["chat"]["default_model"]
    if model_key not in cfg["models"]:
        sys.exit(f"Unknown model '{model_key}'. Options: {', '.join(cfg['models'])}")

    try:
        assistant = Assistant(cfg, model_key, profile)
    except ValueError as e:  # e.g. knowledge base ID not configured yet
        sys.exit(f"Config error: {e}")
    session = Session.load(session_dir, session_id) if session_id else Session(session_dir)

    if assistant.kb:
        mode = f"exposure KB {assistant.kb.id}"
        mode += f" + price agent (KB {assistant.price_agent.kb.id})" if assistant.price_agent else ""
        mode += f" | rerank {cfg['rerank']['model_id']}" if assistant.reranker else " | no rerank"  # improvement 2 ranking step
    else:
        mode = "general knowledge (no knowledge base)"
    print(f"Sydney Art Market Assistant | session {session.id} | model {assistant.model_key} | {mode}")
    print("Type /help for commands.\n")

    while True:
        try:
            text = input(f"[{assistant.model_key}] you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue

        if text.startswith("/"):
            cmd, _, arg = text.partition(" ")
            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/models":
                for k, v in cfg["models"].items():
                    mark = "*" if k == assistant.model_key else " "
                    print(f" {mark} {k}: {v['model_id']}")
            elif cmd == "/model":
                try:
                    assistant.switch(arg.strip())
                    print(f"Switched to {assistant.model_key}")
                except KeyError:
                    print(f"Unknown model. Options: {', '.join(cfg['models'])}")
            elif cmd == "/kb":
                target, question = assistant.kb, arg.strip()
                if question.startswith("price ") or question == "price":
                    target, question = (assistant.price_agent.kb if assistant.price_agent else None), question[5:].strip()
                if not target:
                    print("That knowledge base is disabled (knowledge_base / price_agent in config.yaml).")
                elif not question:
                    print("Usage: /kb [price] <question>")
                else:
                    try:
                        found = target.retrieve(question, min_score=0)
                    except (ClientError, BotoCoreError) as e:
                        print(f"[error] {e}")
                        continue
                    cutoff = target.min_score
                    for i, p in enumerate(found, 1):
                        flag = "" if p.score >= cutoff else "  (below min_score: would be ignored)"
                        # improvement 2 ranking step: show the rerank relevance next to the raw vector similarity
                        vec = f" (vector {p.vector_score:.2f})" if assistant.reranker else ""
                        print(f"[{i}] relevance {p.score:.3f}{vec}  {p.source}{flag}\n    {p.text[:300]!r}\n")
                    print(f"{len(found)} passage(s); min_score = {cutoff} ({'rerank' if assistant.reranker else 'vector'} score)")
            elif cmd == "/history":
                for e in session.log:
                    who = f" / {e.get('agent', 'main')} agent / {e['model']}" if e["role"] == "assistant" else ""
                    print(f"[{e['role']}{who}] {e['text']}\n")
            elif cmd == "/new":
                session = Session(session_dir)
                print(f"New session {session.id}")
            elif cmd == "/sessions":
                print_sessions(session_dir)
            else:
                print("Unknown command. /help")
            continue

        session.add("user", text, assistant.model_key)
        try:
            reply, final_agent = render_reply(assistant, assistant.stream_reply(session))
            if assistant.last_sources:
                print("Sources:")
                for s in assistant.last_sources:
                    print(f"  [{s['id']}] {s['file']} ({s['agent']} agent, score {s['score']:.2f})")
                print()
        except ClientError as e:
            code = e.response["Error"]["Code"]
            hint = " (credentials expired? re-run saml2aws login)" if code in ("ExpiredTokenException", "UnrecognizedClientException") else ""
            print(f"\n[error] {code}: {e.response['Error']['Message']}{hint}\n")
            session.pop_last()
            continue
        except BotoCoreError as e:
            print(f"\n[error] {e}\n")
            session.pop_last()
            continue
        session.add("assistant", reply, assistant.model_key, assistant.last_sources,
                    agent=final_agent, calls=assistant.last_calls)
        session.save()

    session.save() if session.messages else None
    print(f"Session saved: {session.id}")
