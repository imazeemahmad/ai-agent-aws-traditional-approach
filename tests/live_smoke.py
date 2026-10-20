"""Ask the REAL agents questions (needs an AWS login) and show how each one was routed.

    python -m tests.live_smoke                              # one question per route
    python -m tests.live_smoke "What does Fazkurooni cost?" "Who won the AFL?"
    python -m tests.live_smoke --model nova "..."           # try another configured model

For every question it prints the labelled reply, the sources with their agent, the sub-agent calls (question sent and
answer returned) and timing. Nothing is saved to sessions/. For scored, repeatable runs use `python main.py --eval`.
"""
import argparse
import sys
import time

from botocore.exceptions import BotoCoreError, ClientError

from src import chat
from src.assistant import Assistant

DEFAULT_QUESTIONS = [
    "Who won the AFL grand final?",                                                        # expect: [system] refusal, no LLM call
    "What is the price of the Fa Inni Qareeb calligraphy artwork?",                        # expect: routed directly to the price agent
    "What are the stages of commissioning and approvals for public art at Barangaroo?",    # expect: main agent
    "What are the commissioning stages at Barangaroo, and how much does Fazkurooni cost?", # expect: main agent + price agent
]


def main():
    ap = argparse.ArgumentParser(prog="tests.live_smoke", description="Ask the real agents questions and show the routing.")
    ap.add_argument("questions", nargs="*", help="questions to ask (default: one per route)")
    ap.add_argument("--model", help="model key from config.yaml (default: chat.default_model)")
    ap.add_argument("--profile", help="AWS profile (overrides config.yaml aws.profile)")
    ap.add_argument("--config", default=str(chat.CONFIG_PATH))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = chat.load_config(args.config)
    if args.model and args.model not in cfg["models"]:
        sys.exit(f"Unknown model '{args.model}'. Options: {', '.join(cfg['models'])}")
    try:
        assistant = Assistant(cfg, args.model or cfg["chat"]["default_model"], args.profile)
    except ValueError as e:
        sys.exit(f"Config error: {e}")
    print(f"model {assistant.model_key} | exposure KB {assistant.kb.id if assistant.kb else '-'} | "
          f"price KB {assistant.price_agent.kb.id if assistant.price_agent else '-'} | "
          f"rerank {'on' if assistant.reranker else 'off'}\n")

    failures = 0
    for question in args.questions or DEFAULT_QUESTIONS:
        print("=" * 100 + f"\nYOU> {question}")
        session = chat.Session(chat.BASE / "sessions")  # in memory only: never saved
        session.add("user", question, assistant.model_key)
        started = time.perf_counter()
        try:
            _, final_agent = chat.render_reply(assistant, assistant.stream_reply(session))
        except ClientError as e:
            code = e.response["Error"]["Code"]
            hint = "  -> credentials expired? run: saml2aws login --profile sand" if code in (
                "ExpiredTokenException", "UnrecognizedClientException") else ""
            print(f"[error] {code}: {e.response['Error']['Message']}{hint}")
            failures += 1
            continue
        except BotoCoreError as e:
            print(f"[error] {e}")
            failures += 1
            continue
        elapsed = time.perf_counter() - started
        agents = sorted({s["agent"] for s in assistant.last_sources}) or ["none"]
        print(f"   answered by: {final_agent} | sources from: {', '.join(agents)} | {elapsed:.1f}s")
        for s in assistant.last_sources:
            print(f"   [{s['id']}] {s['file']} ({s['agent']}, relevance {s['score']:.2f})")
        for c in assistant.last_calls:
            how = "routed directly" if c["routed"] else "called as a tool"
            print(f"   {c['agent']} agent {how}: asked {c['question']!r}")
    print()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
