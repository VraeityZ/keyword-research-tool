# keyword_research_tool.py
"""
Streamlit application for automated keyword permutation, scoring, and clustering.

FEATURES
    • Text boxes to paste seed topics and modifier lists
    • One click permutation generator
    • Optional Google Ads Keyword Planner or DataForSEO integration for volume and competition metrics
    • Automatic scoring (Volume × (1‑Difficulty))
    • K‑means clustering to group similar phrases
    • Interactive table and CSV download

INSTRUCTIONS
    1) Install requirements: pip install streamlit pandas scikit‑learn requests
    2) run: streamlit run keyword_research_tool.py
    3) Paste lists, click buttons, download CSV

Replace API_KEY with your credential if using DataForSEO. Google Ads API needs OAuth; see documentation links inside code.
"""

import itertools
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# -------------------------------------------
# Helper functions
# -------------------------------------------

def parse_multiline(text: str) -> List[str]:
    """Split textarea input into clean list."""
    return [w.strip() for w in text.splitlines() if w.strip()]


def generate_permutations(seeds: List[str], modifiers: Dict[str, List[str]]) -> List[str]:
    """Cartesian product of seed with at most one element from each active modifier list."""
    active_lists = [mod for mod in modifiers.values() if mod]
    if not active_lists:
        return seeds
    combos = []
    for seed in seeds:
        # Build product across modifier categories
        for combo in itertools.product(*active_lists):
            phrase = " ".join([seed, *combo])
            combos.append(" ".join(phrase.split()))  # assure single spaces
    return combos


def fetch_keyword_metrics(phrases: List[str], engine: str = "mock", api_key: str = "") -> Dict[str, Tuple[int, float]]:
    """Return search volume and competition difficulty for each phrase.
    Placeholder uses mock numbers when engine == 'mock'.
    For DataForSEO: set engine='dataforseo' and supply api_key.
    For Google Ads: integrate with Google Ads API (not implemented here).
    """
    metrics = {}
    if engine == "mock":
        import random
        for p in phrases:
            vol = random.randint(10, 5000)
            diff = round(random.uniform(0.1, 0.9), 2)
            metrics[p] = (vol, diff)
    elif engine == "dataforseo":
        headers = {"Content-Type": "application/json", "Authorization": f"Basic {api_key}"}
        payload = [{"keyword": p, "language_code": "en", "location_code": 2840} for p in phrases]
        resp = requests.post("https://api.dataforseo.com/v3/keywords_data/google_ads/keywords_for_keywords/live", headers=headers, data=json.dumps(payload))
        data = resp.json()
        for kw_data in data.get("tasks", [{}])[0].get("result", []):
            key = kw_data.get("keyword", "")
            vol = kw_data.get("search_volume", 0)
            diff = kw_data.get("competition", 0)
            metrics[key] = (vol, diff)
    else:
        raise ValueError("Unsupported engine")
    return metrics


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["score"] = (df["volume"] * (1 - df["difficulty"])).round(1)
    return df


def cluster_keywords(phrases: List[str], n_clusters: int = 10) -> List[int]:
    vect = TfidfVectorizer(stop_words="english")
    X = vect.fit_transform(phrases)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels.tolist()

# -------------------------------------------
# Streamlit interface
# -------------------------------------------

st.set_page_config(page_title="Keyword Research Tool", layout="wide")
st.title("Automated Keyword Research")

with st.sidebar:
    st.header("Input Lists")

    seeds_text = st.text_area(
        "Seed topics (one per line)",
        "Alcohol misuse\nCannabis dependency\nOpioid addiction",
        height=120,
    )

    age_text = st.text_area("Age filters", "teen\ncollege\nadult\nsenior", height=100)
    condition_text = st.text_area(
        "Co occurring conditions", "depression\nanxiety\nPTSD\nADHD", height=100
    )
    therapy_text = st.text_area(
        "Therapy approaches", "cognitive behavioural therapy\ndialectical behaviour therapy\nacceptance and commitment therapy", height=120
    )
    format_text = st.text_area(
        "Service format", "online\ntelehealth\nindividual\ngroup", height=100
    )
    phase_text = st.text_area("Severity or phase", "early misuse\nrelapse prevention", height=80)

    fetch_metrics = st.checkbox("Fetch real search metrics (requires API key)")
    engine_option = st.selectbox("Metric source", ["mock", "dataforseo"], index=0)
    api_key_input = st.text_input("API key", type="password") if fetch_metrics else ""
    clusters = st.number_input("Number of clusters", 2, 20, 10)

# Process inputs
seeds = parse_multiline(seeds_text)
modifiers = {
    "age": parse_multiline(age_text),
    "condition": parse_multiline(condition_text),
    "therapy": parse_multiline(therapy_text),
    "format": parse_multiline(format_text),
    "phase": parse_multiline(phase_text),
}

if st.button("Generate permutations"):
    phrases = generate_permutations(seeds, modifiers)
    st.subheader(f"Generated phrases: {len(phrases)}")
    st.write(phrases[:50])  # preview first 50

    # Metrics retrieval
    if fetch_metrics:
        st.info("Fetching search metrics this may take a minute")
        metrics = fetch_keyword_metrics(phrases, engine=engine_option, api_key=api_key_input)
        df = pd.DataFrame([
            {
                "keyword": p,
                "volume": metrics.get(p, (0, 0))[0],
                "difficulty": metrics.get(p, (0, 0))[1],
            }
            for p in phrases
        ])
    else:
        df = pd.DataFrame({"keyword": phrases})
        df["volume"] = 0
        df["difficulty"] = 0.5

    df = compute_scores(df)
    df["cluster"] = cluster_keywords(df["keyword"].tolist(), n_clusters=clusters)

    st.dataframe(df)

    csv_path = Path("keywords.csv")
    df.to_csv(csv_path, index=False)
    st.success("CSV saved to keywords.csv in current directory")
    st.download_button("Download CSV", data=csv_path.read_bytes(), file_name="keywords.csv", mime="text/csv")

    st.caption("Score = volume × (1 minus difficulty)")
