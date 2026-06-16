"""
================================================================================
 File   : src/data_prep/news_loader.py
 Project: Stock Portfolio Optimization — PSX DRL Temporal Encoding
 Purpose: Loads all 3 raw news CSVs (Dawn, Brecorder, Mettis), applies the
          same filters used in the scrapers, merges into one news_processed.csv

 Input  : data/raw/news/dawn_pakistan_raw.csv
          data/raw/news/brecorder_pakistan_raw.csv
          data/raw/news/mettis_pakistan_raw.csv

 Output : data/processed/news/news_processed.csv
          columns: date | source | category | title

 Cache  : If output already exists -> return it directly, skip everything
================================================================================
"""

import os, re, logging
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


# ── Filters (same as scrapers) ────────────────────────────────────────────────

FOREIGN_SIGNALS = (
    r"\b(european[ ]stock|europe[ ]stock|asian[ ]markets|asian[ ]stocks"
    r"|wall[ ]street|nasdaq|dow[ ]jones|s&p[ ]500|ftse|dax|nikkei|hang[ ]seng"
    r"|sensex|nifty|bse[ ]india|nse[ ]india|shanghai|shenzhen|csi[ ]300"
    r"|kospi|straits[ ]times|asx[ ]200|federal[ ]reserve|fed[ ]reserve"
    r"|ecb|bank[ ]of[ ]england|rbi|pboc"
    r"|us[ ]stocks|us[ ]markets|global[ ]markets|world[ ]markets"
    r"|china[ ]stocks|india[ ]stocks|uk[ ]stocks|euro[ ]zone|eurozone"
    r"|yuan|renminbi|yen[ ]falls|yen[ ]rises|euro[ ]falls|euro[ ]rises"
    r"|brent[ ]falls|brent[ ]rises|crude[ ]falls|crude[ ]rises"
    r"|oil[ ]falls|oil[ ]rises|gold[ ]falls|gold[ ]rises"
    r"|dollar[ ]index|dxy|london[ ]stock|new[ ]york[ ]stock|tokyo[ ]stock"
    r"|hong[ ]kong[ ]stock|toronto[ ]stock)\b"
)

EXCLUDE_KEYWORDS = (
    r"\b(bollywood|lollywood|oscar|grammy|filmfare|actor|actress"
    r"|film[ ]review|movie|drama[ ]serial|television[ ]show|reality[ ]show"
    r"|game[ ]show|fashion[ ]week|skin[ ]care|hair[ ]care|make.?up|beauty"
    r"|perfume|recipe|cooking|food[ ]trend|restaurant[ ]review|cafe[ ]review"
    r"|horoscope|zodiac|astrology|numerology|travel[ ]guide|tourism"
    r"|visa[ ]guide|adventure[ ]travel|book[ ]review|novel|poetry|fiction"
    r"|cricket[ ]score|match[ ]report|match[ ]preview|ipl|champions[ ]trophy"
    r"|football[ ]result|hockey[ ]result|tennis[ ]result|golf[ ]result"
    r"|fifa|uefa|nba|nhl|nfl|olympics|commonwealth[ ]games"
    r"|phone[ ]review|laptop[ ]review|gadget[ ]review|gaming|video[ ]game"
    r"|android[ ]update|ios[ ]update|viral[ ]video|tiktok|instagram[ ]reel"
    r"|youtube[ ]star|influencer|meme|social[ ]media[ ]trend"
    r"|weather[ ]forecast|rain[ ]forecast|temperature[ ]record"
    r"|heat[ ]wave[ ]forecast|murder[ ]case|robbery|kidnapping"
    r"|road[ ]accident|drug[ ]haul|drug[ ]bust|dental|hair[ ]loss"
    r"|weight[ ]loss[ ]tip|diet[ ]plan|yoga|meditation)\b"
)

PAKISTAN_ANCHOR = (
    r"\b(pakistan|pakistani"
    r"|karachi|lahore|islamabad|rawalpindi|peshawar|quetta|multan"
    r"|faisalabad|hyderabad|sialkot|sukkur|larkana|mirpur"
    r"|sindh|punjab|balochistan|kpk|khyber"
    r"|psx|kse|nccpl|secp"
    r"|sbp|pkr|rupee|kibor"
    r"|fbr|imf|cpec|sifc"
    r"|ogdc|ppl|pso|sngpl|ssgc|hubco|kapco|nepra|ppib|wapda"
    r"|engro|fauji|mlcf|dgkc|bestway|lucky[ ]cement|maple[ ]leaf"
    r"|mcb|ubl|hbl|nbp|meezan|askari|bank[ ]alfalah|faysal|js[ ]bank"
    r"|indus[ ]motor|pak[ ]suzuki|millat|al[ ]ghazi|ptcl|pia"
    r"|nestle[ ]pakistan|unilever[ ]pakistan|abbott[ ]pakistan|searle"
    r"|gul[ ]ahmed|interloop|bata[ ]pakistan|unity[ ]foods|pioneer[ ]cement"
    r"|cherat|kohat[ ]cement|attock[ ]cement|agritech|fatima[ ]fertilizer"
    r"|dawood[ ]hercules|ici[ ]pakistan|nishat|chenab|sapphire"
    r"|service[ ]industries|sazgar|packages)\b"
)

CATEGORY_PATTERNS = {
    "equities": (
        r"\b(kse|psx|stock[ ]market|share[ ]market|stocks|shares|equit"
        r"|listed[ ]compan|scrip|ipo|initial[ ]public[ ]offer"
        r"|bonus[ ]share|right[ ]share|stock[ ]split|buy[ ]back"
        r"|market[ ]cap|index[ ]point|rally|sell[ ]off|bull[ ]run|bear[ ]market"
        r"|trading[ ]volume|circuit[ ]breaker|upper[ ]lock|lower[ ]lock|nccpl)\b"
    ),
    "macro": (
        r"\b(gdp|gross[ ]domestic|cpi|inflation|consumer[ ]price"
        r"|current[ ]account|trade[ ]deficit|trade[ ]surplus|trade[ ]balance"
        r"|balance[ ]of[ ]payment|bop|foreign[ ]reserve|forex[ ]reserve"
        r"|remittance|worker[ ]remittance|large[ ]scale[ ]manufacturing|lsm"
        r"|economic[ ]growth|economic[ ]contraction|recession|unemployment)\b"
    ),
    "monetary": (
        r"\b(sbp|state[ ]bank|monetary[ ]policy|mpc|policy[ ]rate"
        r"|discount[ ]rate|repo[ ]rate|kibor|karachi[ ]interbank"
        r"|t[ ]bill|treasury[ ]bill|pib|pakistan[ ]investment[ ]bond"
        r"|bond[ ]auction|interest[ ]rate|rate[ ]cut|rate[ ]hike"
        r"|rate[ ]unchanged|liquidity[ ]injection)\b"
    ),
    "forex": (
        r"\b(rupee|pkr|dollar[ ]rate|exchange[ ]rate|interbank[ ]rate"
        r"|open[ ]market[ ]rate|currency[ ]depreciati|currency[ ]appreciati"
        r"|devaluation|revaluation|dollar[ ]shortage|kerb[ ]market"
        r"|hawala|hundi)\b"
    ),
    "fiscal": (
        r"\b(federal[ ]budget|mini[ ]budget|supplementary[ ]budget|fbr"
        r"|tax[ ]revenue|tax[ ]collection|revenue[ ]shortfall|revenue[ ]target"
        r"|fiscal[ ]deficit|primary[ ]deficit|primary[ ]surplus"
        r"|public[ ]debt|domestic[ ]debt|external[ ]debt|debt[ ]to[ ]gdp"
        r"|eurobond|sukuk|privatization|privatisation|psdp"
        r"|subsidy[ ]removal|subsidy[ ]cut|circular[ ]debt"
        r"|adb[ ]loan|world[ ]bank[ ]loan|ecnec|cdwp)\b"
    ),
    "energy": (
        r"\b(ogdc|ppl|pso|sui[ ]northern|sui[ ]southern|sngpl|ssgc"
        r"|hubco|kapco|nepra|ppib|petroleum[ ]levy|petrol[ ]price"
        r"|diesel[ ]price|fuel[ ]price|fuel[ ]adjustment"
        r"|electricity[ ]tariff|power[ ]tariff|tariff[ ]adjustment"
        r"|capacity[ ]payment|ipp|loadshedding|rlng|gas[ ]curtailment)\b"
    ),
    "banking": (
        r"\b(hbl|ubl|mcb|nbp|meezan|bank[ ]alfalah|askari|faysal|js[ ]bank"
        r"|allied[ ]bank|habib[ ]metro|silk[ ]bank"
        r"|non[ ]performing[ ]loan|npl|infection[ ]ratio|capital[ ]adequacy"
        r"|advance[ ]to[ ]deposit|adr|deposit[ ]growth|credit[ ]growth"
        r"|banking[ ]profit|banking[ ]sector)\b"
    ),
    "corporates": (
        r"\b(engro|lucky[ ]cement|dgkc|dg[ ]khan|maple[ ]leaf|mlcf|bestway"
        r"|fauji[ ]cement|fauji[ ]fertilizer|ffbl|fatima[ ]fertilizer"
        r"|dawood[ ]hercules|nestle[ ]pakistan|unilever[ ]pakistan"
        r"|abbott[ ]pakistan|searle|indus[ ]motor|pak[ ]suzuki"
        r"|millat[ ]tractor|al[ ]ghazi|sazgar|gul[ ]ahmed|interloop"
        r"|nishat|sapphire|chenab|packages"
        r"|annual[ ]result|quarterly[ ]result|eps|earnings[ ]per[ ]share"
        r"|dividend[ ]announced|dividend[ ]declared|profit[ ]after[ ]tax"
        r"|profit[ ]before[ ]tax|topline|bottomline)\b"
    ),
    "commodities": (
        r"\b(urea[ ]price|dap[ ]price|fertilizer[ ]price|cotton[ ]price"
        r"|cotton[ ]export|wheat[ ]procurement|wheat[ ]support[ ]price"
        r"|sugar[ ]price|sugar[ ]mill|palm[ ]oil[ ]import|edible[ ]oil[ ]pakistan"
        r"|gold[ ]price[ ]pakistan|gold[ ]rate[ ]pakistan|gold[ ]rate[ ]today"
        r"|cement[ ]dispatches|cement[ ]offtake|cement[ ]export)\b"
    ),
    "market_political": (
        r"\b(imf[ ]condition|imf[ ]demand|imf[ ]benchmark|imf[ ]deadline"
        r"|imf[ ]board|imf[ ]approval|imf[ ]disbursement|imf[ ]tranche"
        r"|imf[ ]review|imf[ ]program|imf[ ]bailout|imf[ ]mission|imf[ ]staff"
        r"|fatf[ ]grey|fatf[ ]black|fatf[ ]plenary|fatf[ ]action"
        r"|credit[ ]rating[ ]pakistan|rating[ ]downgrade|rating[ ]upgrade"
        r"|moody|fitch|cpec[ ]investment|cpec[ ]project|cpec[ ]corridor"
        r"|saudi[ ]deposit|uae[ ]deposit|china[ ]swap|bilateral[ ]swap"
        r"|privatization[ ]commission|martial[ ]law)\b"
    ),
}

# standard 5 categories
CATEGORY_MAP = {
    "macro"            : "macro",
    "fiscal"           : "macro",
    "monetary"         : "macro",
    "market_political" : "macro",
    "general_market"   : "macro",
    "general"          : "macro",
    "rates"            : "macro",
    "economy"          : "macro",
    "corporates"       : "corporate",
    "equities"         : "corporate",
    "stocks"           : "corporate",
    "corporate"        : "corporate",
    "equity"           : "corporate",
    "company_analysis" : "corporate",
    "technical"        : "corporate",
    "analyst_briefing" : "corporate",
    "stock_picks"      : "corporate",
    "press_release"    : "corporate",
    "native"           : "corporate",
    "mg_opinion"       : "corporate",
    "energy"           : "energy",
    "commodities"      : "energy",
    "forex"            : "forex",
    "exchange"         : "forex",
    "banking"          : "banking",
}


# ── Filter helpers ────────────────────────────────────────────────────────────

def should_keep(title: str, description: str = "", api_category: str = "") -> bool:
    text = f"{title} {description}".strip()
    if not title or len(title) < 15:
        return False
    if re.search(FOREIGN_SIGNALS, text, re.IGNORECASE):
        return False
    if re.search(EXCLUDE_KEYWORDS, text, re.IGNORECASE):
        return False
    if api_category == "global_business":
        return False
    if not re.search(PAKISTAN_ANCHOR, text, re.IGNORECASE):
        return False
    for pattern in CATEGORY_PATTERNS.values():
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def get_category(title: str, description: str = "") -> str:
    text = f"{title} {description}".strip()
    cats = [c for c, p in CATEGORY_PATTERNS.items()
            if re.search(p, text, re.IGNORECASE)]
    raw  = cats[0] if cats else "general_market"
    return CATEGORY_MAP.get(raw, "macro")


def _normalize(title: str) -> str:
    title = str(title).lower().strip()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


# ── Loaders per source ────────────────────────────────────────────────────────

def _load_dawn(path, train_start, test_end):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[(df["date"] >= train_start) & (df["date"] <= test_end)].copy()
    df["description"] = ""
    mask = df.apply(lambda r: should_keep(str(r["title"])), axis=1)
    df   = df[mask].copy()
    df["category"] = df["title"].apply(lambda t: get_category(t))
    df["source"]   = "dawn"
    log.info("Dawn:      %d rows after filter", len(df))
    return df[["date", "source", "category", "title"]]


def _load_brecorder(path, train_start, test_end):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[(df["date"] >= train_start) & (df["date"] <= test_end)].copy()
    mask = df.apply(lambda r: should_keep(str(r["title"])), axis=1)
    df   = df[mask].copy()
    df["category"] = df["title"].apply(lambda t: get_category(t))
    df["source"]   = "brecorder"
    log.info("Brecorder: %d rows after filter", len(df))
    return df[["date", "source", "category", "title"]]


def _load_mettis(path, train_start, test_end):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[(df["date"] >= train_start) & (df["date"] <= test_end)].copy()
    df["description"] = df["description"].fillna("")
    mask = df.apply(
        lambda r: should_keep(str(r["title"]), str(r["description"]), str(r.get("category", ""))),
        axis=1
    )
    df   = df[mask].copy()
    df["category"] = df.apply(
        lambda r: get_category(str(r["title"]), str(r["description"])), axis=1
    )
    df["source"] = "mettis"
    # mettis: combine title + description for richer text
    df["title"] = (df["title"].fillna("") + " " + df["description"].fillna("")).str.strip()
    log.info("Mettis:    %d rows after filter", len(df))
    return df[["date", "source", "category", "title"]]


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(df):
    SOURCE_PRIORITY = {"brecorder": 0, "dawn": 1, "mettis": 2}
    before = len(df)
    df = df.copy()
    df["_norm"]     = df["title"].fillna("").apply(_normalize)
    df["_priority"] = df["source"].map(SOURCE_PRIORITY).fillna(99)
    df = df.sort_values(["date", "_norm", "_priority"])
    df = df.drop_duplicates(subset=["date", "_norm"], keep="first")
    df = df.drop_duplicates(subset=["_norm"],         keep="first")
    df = df.drop(columns=["_norm", "_priority"]).reset_index(drop=True)
    log.info("Dedup: %d -> %d rows (removed %d)", before, len(df), before - len(df))
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def run(cfg=None):
    if cfg is None:
        cfg = load_config()

    raw_news_dir   = _resolve(cfg["data"]["raw_news_dir"])
    processed_dir  = _resolve(cfg["data"]["processed_news_dir"])
    output_path    = os.path.join(processed_dir, "news_processed.csv")
    train_start    = cfg["data"]["train_start"]
    test_end       = cfg["data"]["test_end"]

    os.makedirs(processed_dir, exist_ok=True)

    # cache check
    if os.path.exists(output_path):
        log.info("Cache hit -> %s", output_path)
        return pd.read_csv(output_path, parse_dates=["date"])

    dawn_path      = os.path.join(raw_news_dir, "dawn_pakistan_raw.csv")
    brecorder_path = os.path.join(raw_news_dir, "brecorder_pakistan_raw.csv")
    mettis_path    = os.path.join(raw_news_dir, "mettis_pakistan_raw.csv")

    dfs = []
    for path, loader in [
        (dawn_path,      _load_dawn),
        (brecorder_path, _load_brecorder),
        (mettis_path,    _load_mettis),
    ]:
        if not os.path.exists(path):
            log.warning("Missing: %s — skipping", path)
            continue
        dfs.append(loader(path, train_start, test_end))

    if not dfs:
        raise FileNotFoundError("No raw news CSVs found.")

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["date", "title"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info("Combined: %d rows", len(df))

    df = _deduplicate(df)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df[["date", "source", "category", "title"]]

    df.to_csv(output_path, index=False)
    log.info("Saved -> %s  (%d rows)", output_path, len(df))
    return df


if __name__ == "__main__":
    run()
