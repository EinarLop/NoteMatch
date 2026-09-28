"""Command line: notematch ingest | recommend "<prompt>" | eval | tune | serve"""

import argparse
import logging
import os

from notematch.config import load_config


def prepare(cfg) -> None:
    """Make a fresh machine ready to serve: pull missing models, build the index if absent."""
    import ollama
    from notematch.ingest import build_index

    tagged = lambda name: name if ":" in name else f"{name}:latest"
    have = {m.model for m in ollama.list().models}
    for name in (cfg.llm.model, cfg.embedding.model):
        if cfg.llm.provider != "ollama" and name == cfg.llm.model:
            continue                                     # API LLM: nothing to pull
        if tagged(name) not in have:
            print(f"Pulling {name} (first run only)...", flush=True)
            ollama.pull(name)
    if not (cfg.paths.index / "manifest.json").exists():
        print("No index found: building it (first run only, ~2 min)...", flush=True)
        build_index(cfg)


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
    tu = sub.add_parser("tune", help="grid-search tau_topic x tau_min on the dev split")
    tu.add_argument("cases", nargs="?", default="eval/cases.jsonl")
    sv = sub.add_parser("serve", help="run the HTTP API (POST /recommend)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--prepare", action="store_true",
                    help="first pull missing models and build the index if absent (Docker entrypoint)")
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
            p = rec_.product
            print(f"{n}. {p.title}  (score {rec_.score}, ${p.price})  {p.purchase_url}")
            for m in rec_.covered_topics:
                print(f"   - {m.topic}: " + "; ".join(f"{s.text} ({s.similarity})" for s in m.sections))
        print(f"\n{r.message}")
    elif args.cmd == "eval":
        from pathlib import Path
        from notematch.evaluate import evaluate
        logging.getLogger("notematch").setLevel(logging.WARNING)   # skip per-request log lines
        name = Path(args.config or os.environ.get("NOTEMATCH_CONFIG", "config.toml")).stem
        evaluate(cfg, Path(args.cases), args.split, name)
    elif args.cmd == "tune":
        from pathlib import Path
        from notematch.evaluate import tune
        logging.getLogger("notematch").setLevel(logging.WARNING)
        tune(cfg, Path(args.cases))
    elif args.cmd == "serve":
        import uvicorn
        if args.config:
            os.environ["NOTEMATCH_CONFIG"] = args.config   # the app loads its config at startup
        if args.prepare:
            prepare(cfg)
        uvicorn.run("notematch.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
