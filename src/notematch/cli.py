"""Command line: notematch ingest | recommend "<prompt>" | eval  (serve comes later)."""

import argparse
import logging
import os

from notematch.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(prog="notematch")
    ap.add_argument("--config", help="config file (default: $NOTEMATCH_CONFIG or config.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="build index/ from the catalog")
    rec = sub.add_parser("recommend", help="recommend notes for a prompt")
    rec.add_argument("prompt")
    ev = sub.add_parser("eval", help="score the pipeline on the eval set")
    ev.add_argument("cases", nargs="?", default="eval/cases.jsonl")
    ev.add_argument("--split", choices=["dev", "test", "all"], default="dev",
                    help="default dev; run test only once, with final thresholds (design 7.3)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")  # request log -> stderr
    logging.getLogger("httpx").setLevel(logging.WARNING)           # ollama client's per-request noise
    cfg = load_config(args.config)

    if args.cmd == "ingest":
        from notematch.ingest import build_index
        build_index(cfg)
    elif args.cmd == "recommend":
        from notematch.recommend import load_index, recommend
        r = recommend(args.prompt, load_index(cfg), cfg)
        print(f"Topics: {r.parsed_query.topics if r.parsed_query else '(parse failed)'}\n")
        for n, rec_ in enumerate(r.recommendations, 1):
            print(f"{n}. {rec_.title}  (score {rec_.score}, ${rec_.price})  {rec_.purchase_url}")
            for m in rec_.covered_topics:
                print(f"   - {m.topic}: " + "; ".join(f"{s.text} ({s.similarity})" for s in m.sections))
        print(f"\n{r.message}")
    elif args.cmd == "eval":
        from pathlib import Path
        from notematch.evaluate import evaluate
        logging.getLogger("notematch").setLevel(logging.WARNING)   # skip per-request log lines
        name = Path(args.config or os.environ.get("NOTEMATCH_CONFIG", "config.toml")).stem
        evaluate(cfg, Path(args.cases), args.split, name)


if __name__ == "__main__":
    main()
