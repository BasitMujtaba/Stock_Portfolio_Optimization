
"""
================================================================================
 File   : main.py
 Project: Stock Portfolio Optimization — PSX DRL Temporal Encoding
 Purpose: Full pipeline orchestrator with CLI stage control.

 Usage:
   python main.py                  # full pipeline (caches skip done stages)
   python main.py --skip-data      # skip data prep + sentiment + build_dataset
   python main.py --skip-train     # skip training (assumes checkpoints exist)
   python main.py --skip-eval      # skip evaluation
   python main.py --skip-ablation  # skip ablation study
   python main.py --only-data      # data pipeline only
   python main.py --only-train     # training only
   python main.py --only-eval      # evaluation only
   python main.py --only-ablation  # ablation only
================================================================================
"""

import os
import sys
import time
import logging
import argparse
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PSX Portfolio Optimization — Full Pipeline"
    )

    skip = parser.add_argument_group("skip flags (run everything EXCEPT these)")
    skip.add_argument("--skip-data",      action="store_true", help="Skip data prep + sentiment + build_dataset")
    skip.add_argument("--skip-train",     action="store_true", help="Skip agent training")
    skip.add_argument("--skip-eval",      action="store_true", help="Skip evaluation")
    skip.add_argument("--skip-ablation",  action="store_true", help="Skip ablation study")

    only = parser.add_argument_group("only flags (run ONLY this stage)")
    only.add_argument("--only-data",      action="store_true", help="Run data pipeline only")
    only.add_argument("--only-train",     action="store_true", help="Run training only")
    only.add_argument("--only-eval",      action="store_true", help="Run evaluation only")
    only.add_argument("--only-ablation",  action="store_true", help="Run ablation only")

    args = parser.parse_args()

    only_flags = [args.only_data, args.only_train, args.only_eval, args.only_ablation]
    skip_flags = [args.skip_data, args.skip_train, args.skip_eval, args.skip_ablation]

    if sum(only_flags) > 1:
        parser.error("Only one --only-* flag may be set at a time.")
    if any(only_flags) and any(skip_flags):
        parser.error("--only-* and --skip-* flags cannot be combined.")

    return args


def _resolve_stages(args):
    stages = {
        "data"     : True,
        "train"    : True,
        "eval"     : True,
        "ablation" : True,
    }

    if args.only_data:
        return {k: k == "data"     for k in stages}
    if args.only_train:
        return {k: k == "train"    for k in stages}
    if args.only_eval:
        return {k: k == "eval"     for k in stages}
    if args.only_ablation:
        return {k: k == "ablation" for k in stages}

    if args.skip_data:
        stages["data"] = False
    if args.skip_train:
        stages["train"] = False
    if args.skip_eval:
        stages["eval"] = False
    if args.skip_ablation:
        stages["ablation"] = False

    return stages


# ── Stage runners ──────────────────────────────────────────────────────────────

def _banner(title: str):
    log.info("")
    log.info("=" * 70)
    log.info("  STAGE: %s", title)
    log.info("=" * 70)


def run_data(cfg):
    _banner("DATA PIPELINE")

    from src.data_prep import price_loader, news_loader, build_dataset
    from src.sentiment import finbert

    t0 = time.time()

    log.info("[1/4] Price loader ...")
    price_loader.run(cfg)

    log.info("[2/4] News loader ...")
    news_loader.run(cfg)

    log.info("[3/4] FinBERT sentiment scoring ...")
    finbert.run(cfg)

    log.info("[4/4] Build features dataset ...")
    build_dataset.run(cfg)

    log.info("DATA PIPELINE done in %.1fs", time.time() - t0)


def run_train(cfg):
    _banner("TRAINING")
    from src.train import main as train_main
    t0 = time.time()
    summary = train_main(cfg)
    log.info("TRAINING done in %.1fs", time.time() - t0)
    return summary


def run_eval(cfg):
    _banner("EVALUATION")
    from src.evaluate import main as eval_main
    t0 = time.time()
    result = eval_main(cfg)
    log.info("EVALUATION done in %.1fs", time.time() - t0)
    return result


def run_ablation(cfg):
    _banner("ABLATION STUDY")
    from src.ablation import main as ablation_main
    t0 = time.time()
    result = ablation_main(cfg)
    log.info("ABLATION done in %.1fs", time.time() - t0)
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    stages = _resolve_stages(args)

    log.info("")
    log.info("=" * 70)
    log.info("  PSX Portfolio Optimization — DRL Temporal Encoding")
    log.info("=" * 70)
    log.info("Stage plan:")
    for stage, active in stages.items():
        status = "RUN " if active else "SKIP"
        log.info("  %-12s %s", stage, status)
    log.info("")

    cfg        = load_config()
    pipeline_t = time.time()

    if stages["data"]:
        run_data(cfg)

    if stages["train"]:
        if not stages["data"]:
            features_path = os.path.join(PROJECT_ROOT, cfg["data"]["features_path"])
            if not os.path.exists(features_path):
                log.error(
                    "features.parquet not found at %s. "
                    "Run with --only-data first.", features_path
                )
                sys.exit(1)
        run_train(cfg)

    if stages["eval"]:
        checkpoint_dir = os.path.join(PROJECT_ROOT, "results", "checkpoints")
        checkpoints    = [
            os.path.join(checkpoint_dir, f"{a}_best.pt")
            for a in ["ddpg", "ppo", "a2c"]
        ]
        if not any(os.path.exists(c) for c in checkpoints):
            log.error(
                "No checkpoints found in %s. "
                "Run with --only-train first.", checkpoint_dir
            )
            sys.exit(1)
        run_eval(cfg)

    if stages["ablation"]:
        features_path = os.path.join(PROJECT_ROOT, cfg["data"]["features_path"])
        if not os.path.exists(features_path):
            log.error(
                "features.parquet not found. "
                "Run data pipeline before ablation."
            )
            sys.exit(1)
        run_ablation(cfg)

    log.info("")
    log.info("=" * 70)
    log.info("  PIPELINE COMPLETE in %.1fs", time.time() - pipeline_t)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
