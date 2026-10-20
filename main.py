"""Entry point: defines the command-line parser and starts the chat."""
import argparse

from src import chat


def build_parser():
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="Sydney artwork market exposure assistant (AWS Bedrock, CLI POC).",
    )
    ap.add_argument("--config", default=str(chat.CONFIG_PATH), help="path to config file (default: config.yaml)")
    ap.add_argument("--profile", help="AWS profile (overrides config.yaml aws.profile)")
    ap.add_argument("--model", help="model key from config.yaml (default: chat.default_model)")
    ap.add_argument("--session", help="resume a saved session id")
    ap.add_argument("--list-models", action="store_true", help="list configured models and exit")
    ap.add_argument("--list-sessions", action="store_true", help="list saved sessions and exit")
    # improvement 2 ranking step: choose min_score per knowledge base from labelled questions
    ap.add_argument("--calibrate", nargs="?", const="eval/relevance_set.yaml", metavar="FILE",
                    help="score labelled in/out-of-scope questions against the KBs, suggest min_score, and exit")
    # evaluation harness (src/evaluate.py): score an implementation on eval/cases.yaml
    ap.add_argument("--eval", nargs="?", const="eval/cases.yaml", metavar="FILE",
                    help="run the labelled evaluation cases, print a score, save eval/results/<time>-<label>.json, and exit")
    ap.add_argument("--eval-label", default="run", help="name for this evaluation run (used in the result file name)")
    ap.add_argument("--eval-repeat", type=int, default=1, metavar="N",
                    help="run each case N times: LLM answers vary, so N>=3 shows flaky cases (default 1)")
    ap.add_argument("--eval-filter", metavar="TEXT", help="only cases whose id or category contains TEXT")
    ap.add_argument("--eval-target", default="assistant", help="implementation to score (see TARGETS in src/evaluate.py)")
    ap.add_argument("--eval-compare", nargs=2, metavar=("A.json", "B.json"), help="compare two saved results and exit")
    return ap


def main():
    args = build_parser().parse_args()
    if args.eval_compare:
        from src import evaluate
        evaluate.compare(*args.eval_compare)
        return
    if args.eval:
        from src import evaluate
        evaluate.run(args.config, args.profile, chat.BASE / args.eval, args.eval_label, args.eval_repeat,
                     args.eval_filter, args.eval_target)
        return
    if args.calibrate:  # improvement 2 ranking step
        from src import calibrate
        calibrate.run(args.config, args.profile, chat.BASE / args.calibrate)
        return
    if args.list_models or args.list_sessions:
        cfg = chat.load_config(args.config)
        if args.list_models:
            for key, m in cfg["models"].items():
                print(f"  {key}: {m['model_id']}")
        if args.list_sessions:
            chat.print_sessions(chat.BASE / cfg["chat"]["session_dir"])
        return
    chat.run(args.config, args.profile, args.model, args.session)


if __name__ == "__main__":
    main()
