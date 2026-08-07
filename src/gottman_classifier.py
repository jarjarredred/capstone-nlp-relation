"""Gottman's Four Horsemen Relationship Conflict Detector

Capstone NLP Relation Pipeline
Author: Jarred R. Gastreich
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


# =====================================================================
# 1. DATA LOADING & TEMPORAL FEATURE ENGINEERING
# =====================================================================
def load_and_prepare_data(file_path: Path) -> pd.DataFrame:
    """Loads cleaned CallHome dataset and constructs temporal and lag metrics."""
    print(f"📥 Loading clean dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # Calculate turn duration in seconds
    df["turn_duration"] = (df["end_ms"] - df["start_ms"]) / 1000.0

    # Group-based lag features to calculate cross-turn response latency
    df["prev_speaker"] = df.groupby("file_id")["speaker_raw"].shift(1)
    df["prev_end_ms"] = df.groupby("file_id")["end_ms"].shift(1)

    # Response latency (ms & seconds) between alternating speaker turns
    df["response_latency_ms"] = np.where(
        (df["speaker_raw"] != df["prev_speaker"]) & (df["prev_speaker"].notna()),
        df["start_ms"] - df["prev_end_ms"],
        np.nan,
    )
    df["response_latency_s"] = df["response_latency_ms"] / 1000.0

    # Impute missing conversation-start latencies with dataset median
    df["response_latency_s"] = df["response_latency_s"].fillna(
        df["response_latency_s"].median()
    )

    return df


# =====================================================================
# 2. LEXICAL & CONTEXT-AWARE FEATURE EXTRACTION
# =====================================================================
def count_exact_word(text: str, word: str) -> int:
    """Counts whole-word occurrences using regex word boundaries."""
    return len(re.findall(r"\b" + re.escape(word) + r"\b", str(text).lower()))


def extract_gottman_lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts lexical and context-aware indicators mapped to Gottman's Four Horsemen."""
    print("🧠 Extracting context-aware Gottman lexical & temporal indicators...")
    df["text_clean"] = df["text"].fillna("").astype(str).str.lower()

    # Pronoun counts
    df["count_you"] = df["text_clean"].apply(lambda x: count_exact_word(x, "you"))
    df["count_i"] = df["text_clean"].apply(lambda x: count_exact_word(x, "i"))

    # Pronoun density (Criticism indicator)
    df["you_ratio"] = np.where(
        df["word_count"] > 0, df["count_you"] / df["word_count"], 0.0
    )

    # Contrastive Starters (Defensiveness indicator)
    df["starts_contrastive"] = (
        df["text_clean"]
        .str.strip()
        .apply(lambda x: int(x.startswith("but") or x.startswith("however")))
    )

    # Dismissive / Disdain Vocabulary (Contempt indicator)
    contempt_words = [
        "whatever",
        "pfft",
        "sigh",
        "obviously",
        "seriously",
        "ridiculous",
    ]
    df["count_contempt"] = df["text_clean"].apply(
        lambda x: sum(count_exact_word(x, w) for w in contempt_words)
    )
    df["has_contempt_lexicon"] = (df["count_contempt"] > 0).astype(int)

    # --- BASE HORSEMEN FLAGS (CRITICISM, DEFENSIVENESS, CONTEMPT) ---
    df["flag_criticism"] = (df["you_ratio"] > 0.20).astype(int)
    df["flag_defensiveness"] = (df["starts_contrastive"] == 1).astype(int)
    df["flag_contempt"] = (df["has_contempt_lexicon"] == 1).astype(int)

    # --- REFINED CONTEXT-AWARE STONEWALLING LOGIC ---
    minimal_tokens = {
        "yeah",
        "yep",
        "uh-huh",
        "ok",
        "okay",
        "fine",
        "whatever",
        "mhm",
        "sure",
        "right",
    }
    df["is_minimal_token"] = (
        df["text_clean"].str.strip().isin(minimal_tokens).astype(int)
    )

    # Track if the partner attacked on the immediately preceding turn
    df["prev_criticism"] = df.groupby("file_id")["flag_criticism"].shift(1).fillna(0)
    df["prev_contempt"] = df.groupby("file_id")["flag_contempt"].shift(1).fillna(0)
    df["partner_attacked_prev"] = (
        (df["prev_criticism"] == 1) | (df["prev_contempt"] == 1)
    ).astype(int)

    # Context-Aware Stonewalling Flag
    df["flag_stonewalling"] = (
        # Condition A: Extended delay (>3.5s) combined with a minimal token
        ((df["response_latency_s"] > 3.5) & (df["is_minimal_token"] == 1))
        |
        # Condition B: Minimal response/collapse immediately following a partner attack
        ((df["partner_attacked_prev"] == 1) & (df["word_count"] <= 2))
        |
        # Condition C: Severe non-verbal pause (>5.0s delay)
        (df["response_latency_s"] > 5.0)
    ).astype(int)

    # Combined Target Variable: 1 if ANY Horseman is triggered on this turn
    df["is_conflict_turn"] = (
        (df["flag_criticism"] == 1)
        | (df["flag_defensiveness"] == 1)
        | (df["flag_stonewalling"] == 1)
        | (df["flag_contempt"] == 1)
    ).astype(int)

    return df


# =====================================================================
# 3. MACHINE LEARNING MODEL TRAINING & EVALUATION
# =====================================================================
def train_and_evaluate_model(df: pd.DataFrame):
    """Executes group-based train/test split, trains Random Forest, and prints diagnostics."""
    feature_cols = [
        "word_count",
        "turn_duration",
        "response_latency_s",
        "you_ratio",
        "count_you",
        "count_i",
        "starts_contrastive",
        "has_contempt_lexicon",
        "is_minimal_token",
        "partner_attacked_prev",
    ]

    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    y = df["is_conflict_turn"]
    groups = df["file_id"]

    # 80/20 Group-based Train/Test Split (prevents dialogue leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"\n📊 Training Set Size : {len(X_train):,} turns")
    print(f"📊 Test Set Size     : {len(X_test):,} turns")
    print(f"📊 Target Imbalance  : {y.mean() * 100:.2f}% Conflict Turns")

    # Train Random Forest Classifier with Class Balance Reweighting
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)

    # Predictions & Probabilities
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    # Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 60)
    print("🏆 GOTTMAN CONFLICT DETECTOR — MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("=" * 60)

    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Neutral", "Conflict"]))

    # --- Four Horsemen Summary Breakdown ---
    print("=" * 60)
    print("🐎 FOUR HORSEMEN BREAKDOWN ACROSS ENTIRE CORPUS")
    print("=" * 60)
    print(
        f"1. Criticism (You-Ratio > 20%)        : {df['flag_criticism'].sum():>5,} turns ({df['flag_criticism'].mean() * 100:.2f}%)"
    )
    print(
        f"2. Defensiveness (Contrastive Start) : {df['flag_defensiveness'].sum():>5,} turns ({df['flag_defensiveness'].mean() * 100:.2f}%)"
    )
    print(
        f"3. Stonewalling (Contextual Delay)   : {df['flag_stonewalling'].sum():>5,} turns ({df['flag_stonewalling'].mean() * 100:.2f}%)"
    )
    print(
        f"4. Contempt (Disdain Lexicon)        : {df['flag_contempt'].sum():>5,} turns ({df['flag_contempt'].mean() * 100:.2f}%)"
    )
    print("-" * 60)
    print(
        f"Total Identified Conflict Turns      : {df['is_conflict_turn'].sum():>5,} turns ({df['is_conflict_turn'].mean() * 100:.2f}%)"
    )
    print("=" * 60)

    # Plot Feature Importances
    importances = pd.Series(
        rf_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    plt.figure(figsize=(9, 4.5))
    sns.barplot(
        x=importances.values,
        y=importances.index,
        hue=importances.index,
        palette="viridis",
        legend=False,
    )
    plt.title("Gottman Four Horsemen Model — Feature Importances", fontweight="bold")
    plt.xlabel("Gini Importance Score")
    plt.tight_layout()
    plt.show()

    return rf_model


# =====================================================================
# 4. MAIN EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "callhome_tabular_clean.csv"

    df_clean = load_and_prepare_data(DATA_PATH)
    df_features = extract_gottman_lexical_features(df_clean)
    model = train_and_evaluate_model(df_features)
