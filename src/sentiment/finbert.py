"""
================================================================================
 File   : src/sentiment/finbert.py
 Project: Stock Portfolio Optimization — PSX DRL Temporal Encoding
 Purpose: Runs FinBERT on daily aggregated news text from news_processed.csv
          and produces a daily sentiment score per date.

 Input  : data/processed/news/news_processed.csv
 Output : data/processed/news/sentiment_scores.csv
          columns: date | sentiment_score (float in [-1, 1])

 Logic  :
   - Group all titles per date into one text batch
   - Run FinBERT (ProsusAI/finbert) on each title individually
   - Average scores per date (positive=+1, negative=-1, neutral=0)
   - Dates with no news get score 0.0
   - Cache: if sentiment_scores.csv exists -> return it directly
================================================================================
"""

import os, logging
import numpy as np
import pandas as pd
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_NAME = "ProsusAI/finbert"
LABEL_MAP  = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

def _resolve(cfg_path):
    if os.path.isabs(cfg_path):
        return cfg_path
    return os.path.join(PROJECT_ROOT, cfg_path)


# ── Cache check ───────────────────────────────────────────────────────────────

def _cache_valid(path):
    if not os.path.exists(path):
        return False, None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty:
            return False, None
        log.info("Cache hit: %s  (%d dates, score range [%.3f, %.3f])",
                 os.path.basename(path), len(df),
                 df["sentiment_score"].min(), df["sentiment_score"].max())
        return True, df
    except Exception as e:
        log.warning("Cache check failed: %s", e)
        return False, None


# ── FinBERT scorer ────────────────────────────────────────────────────────────

class FinBERTScorer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading FinBERT on %s ...", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        log.info("FinBERT loaded")

    def score_batch(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """
        Score a list of texts. Returns a float per text in [-1, 1].
        positive=+1, negative=-1, neutral=0 weighted by softmax probability.
        """
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs  = softmax(logits, dim=-1).cpu().numpy()
            labels = self.model.config.id2label
            for prob in probs:
                score = sum(
                    LABEL_MAP.get(labels[i].lower(), 0.0) * prob[i]
                    for i in range(len(prob))
                )
                scores.append(float(score))
        return scores

    def score_daily(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes news_processed DataFrame (date, source, category, title).
        Returns DataFrame with columns: date, sentiment_score.
        One row per date. Score = mean of all title scores that day.
        """
        news_df = news_df.copy()
        news_df["date"] = pd.to_datetime(news_df["date"])
        dates   = sorted(news_df["date"].unique())
        results = []

        log.info("Scoring %d unique dates ...", len(dates))
        for date in tqdm(dates, desc="FinBERT"):
            titles = news_df[news_df["date"] == date]["title"].dropna().tolist()
            titles = [t for t in titles if isinstance(t, str) and len(t.strip()) > 5]
            if not titles:
                results.append({"date": date, "sentiment_score": 0.0})
                continue
            day_scores = self.score_batch(titles)
            results.append({
                "date"            : date,
                "sentiment_score" : float(np.mean(day_scores))
            })

        df = pd.DataFrame(results)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df


# ── Fill missing dates ────────────────────────────────────────────────────────

def _fill_missing_dates(sentiment_df, prices_df):
    """
    Ensure every trading date in prices has a sentiment score.
    Dates with no news are filled with 0.0.
    """
    all_dates      = pd.to_datetime(prices_df["date"].unique())
    sentiment_df   = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    missing_dates  = set(all_dates) - set(sentiment_df["date"])

    if missing_dates:
        log.info("Filling %d trading dates with no news -> 0.0", len(missing_dates))
        filler = pd.DataFrame({
            "date"            : sorted(missing_dates),
            "sentiment_score" : 0.0
        })
        sentiment_df = pd.concat([sentiment_df, filler], ignore_index=True)

    sentiment_df = sentiment_df.sort_values("date").reset_index(drop=True)
    sentiment_df["date"] = sentiment_df["date"].dt.strftime("%Y-%m-%d")
    return sentiment_df


# ── Public API ────────────────────────────────────────────────────────────────

def run(cfg=None):
    if cfg is None:
        cfg = load_config()

    news_path      = os.path.join(_resolve(cfg["data"]["processed_news_dir"]), "news_processed.csv")
    prices_path    = os.path.join(_resolve(cfg["data"]["processed_prices_dir"]), "psx_prices_processed.csv")
    output_path    = os.path.join(_resolve(cfg["data"]["processed_news_dir"]), "sentiment_scores.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # cache check
    valid, cached = _cache_valid(output_path)
    if valid:
        return cached

    # load news
    if not os.path.exists(news_path):
        raise FileNotFoundError(f"news_processed.csv not found at {news_path}. Run news_loader.py first.")
    news_df = pd.read_csv(news_path)
    log.info("News loaded: %d rows", len(news_df))

    # load prices for date alignment
    prices_df = pd.read_csv(prices_path, usecols=["date"])
    log.info("Price dates loaded: %d unique trading days", prices_df["date"].nunique())

    # score
    scorer       = FinBERTScorer()
    sentiment_df = scorer.score_daily(news_df)

    # fill missing trading dates
    sentiment_df = _fill_missing_dates(sentiment_df, prices_df)

    sentiment_df.to_csv(output_path, index=False)
    log.info("Saved -> %s  (%d dates, score range [%.3f, %.3f])",
             output_path, len(sentiment_df),
             sentiment_df["sentiment_score"].min(),
             sentiment_df["sentiment_score"].max())
    return sentiment_df


if __name__ == "__main__":
    run()
