"""Gottman Four Horsemen Visualization Suite for Counselor Reports

Generates individual file diagnostic plots, corpus-wide heatmaps,
and batch radar profiles for every file_id in the dataset.

Author: Jarred R. Gastreich
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Clean formatting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 10})


# =====================================================================
# 1. FIXED HEATMAP: FOUR HORSEMEN INTENSITY (% OF SPEAKER TURNS)
# =====================================================================
def plot_fixed_heatmap(df: pd.DataFrame, output_dir: Path, top_n: int = 15):
    """Generates an aligned heatmap showing % of speaker turns for each Horseman."""
    pct_cols = [c for c in df.columns if c.endswith("_pct_A") or c.endswith("_pct_B")]
    for col in pct_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    turn_cols = [
        c for c in df.columns if c.endswith("_turns_A") or c.endswith("_turns_B")
    ]
    df["total_horsemen_turns"] = df[turn_cols].sum(axis=1)

    top_files = (
        df.sort_values(by="total_horsemen_turns", ascending=False).head(top_n).copy()
    )

    heatmap_matrix = pd.DataFrame(
        {
            "Crit A (%)": top_files["criticism_pct_A"].values * 100,
            "Crit B (%)": top_files["criticism_pct_B"].values * 100,
            "Def A (%)": top_files["defensiveness_pct_A"].values * 100,
            "Def B (%)": top_files["defensiveness_pct_B"].values * 100,
            "Cont A (%)": top_files["contempt_pct_A"].values * 100,
            "Cont B (%)": top_files["contempt_pct_B"].values * 100,
            "Stone A (%)": top_files["stonewalling_pct_A"].values * 100,
            "Stone B (%)": top_files["stonewalling_pct_B"].values * 100,
        },
        index=[f"File {fid}" for fid in top_files["file_id"]],
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Percentage of Speaker Turns (%)"},
        ax=ax,
        robust=True,
    )

    ax.set_title(
        f"Gottman Four Horsemen Trigger Rates — Top {top_n} Active Conversations",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.ylabel("Conversation File ID", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontweight="bold")
    plt.tight_layout()

    out_file = output_dir / "horsemen_heatmap_fixed.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ Saved fixed heatmap to: {out_file}")


# =====================================================================
# 2. COUNSELOR DIAGNOSTIC: HORSEMEN BREAKDOWN PER SPEAKER
# =====================================================================
def plot_counselor_horsemen_breakdown(
    df: pd.DataFrame, file_id_target: int, output_dir: Path
):
    """Generates a side-by-side bar comparison for a specific file_id."""
    row = df[df["file_id"] == file_id_target]
    if row.empty:
        return

    row = row.iloc[0]

    horsemen = ["Defensiveness", "Criticism", "Contempt", "Stonewalling"]
    turns_A = [
        row["defensiveness_turns_A"],
        row["criticism_turns_A"],
        row["contempt_turns_A"],
        row["stonewalling_turns_A"],
    ]
    turns_B = [
        row["defensiveness_turns_B"],
        row["criticism_turns_B"],
        row["contempt_turns_B"],
        row["stonewalling_turns_B"],
    ]

    pct_A = [
        row["defensiveness_pct_A"] * 100,
        row["criticism_pct_A"] * 100,
        row["contempt_pct_A"] * 100,
        row["stonewalling_pct_A"] * 100,
    ]
    pct_B = [
        row["defensiveness_pct_B"] * 100,
        row["criticism_pct_B"] * 100,
        row["contempt_pct_B"] * 100,
        row["stonewalling_pct_B"] * 100,
    ]

    x = np.arange(len(horsemen))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute Turn Counts
    rects1 = ax1.bar(x - width / 2, turns_A, width, label="Speaker A", color="#2b5c8f")
    rects2 = ax1.bar(x + width / 2, turns_B, width, label="Speaker B", color="#d95f02")

    ax1.set_ylabel("Number of Identified Turns")
    ax1.set_title(
        f"Conversation {file_id_target}: Absolute Horsemen Turn Count",
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(horsemen, fontweight="bold")
    ax1.legend(frameon=True)

    for rect in rects1 + rects2:
        h = rect.get_height()
        if h > 0:
            ax1.annotate(
                f"{int(h)}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Relative Behavioral Intensity (%)
    rects3 = ax2.bar(x - width / 2, pct_A, width, label="Speaker A", color="#2b5c8f")
    rects4 = ax2.bar(x + width / 2, pct_B, width, label="Speaker B", color="#d95f02")

    ax2.set_ylabel("% of Speaker's Total Turns")
    ax2.set_title(
        f"Conversation {file_id_target}: Relative Behavioral Intensity (%)",
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(horsemen, fontweight="bold")
    ax2.legend(frameon=True)

    for rect in rects3 + rects4:
        h = rect.get_height()
        if h > 0:
            ax2.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    out_file = output_dir / f"counselor_report_file_{file_id_target}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()


# =====================================================================
# 3. RADAR PROFILE GENERATOR (SINGLE FILE)
# =====================================================================
def plot_horsemen_radar_profile(row: pd.Series, output_dir: Path):
    """Generates a 4-axis spider/radar plot for a single file_id row."""
    file_id = int(row["file_id"])
    categories = ["Defensiveness", "Criticism", "Contempt", "Stonewalling"]
    N = len(categories)

    # Values in percentage of turns
    values_A = [
        row["defensiveness_pct_A"] * 100,
        row["criticism_pct_A"] * 100,
        row["contempt_pct_A"] * 100,
        row["stonewalling_pct_A"] * 100,
    ]
    values_B = [
        row["defensiveness_pct_B"] * 100,
        row["criticism_pct_B"] * 100,
        row["contempt_pct_B"] * 100,
        row["stonewalling_pct_B"] * 100,
    ]

    # Close polygon loops
    values_A += values_A[:1]
    values_B += values_B[:1]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    plt.xticks(angles[:-1], categories, color="black", size=11, fontweight="bold")

    # Plot Speaker A & B
    ax.plot(
        angles,
        values_A,
        linewidth=2,
        linestyle="solid",
        label="Speaker A",
        color="#2b5c8f",
    )
    ax.fill(angles, values_A, color="#2b5c8f", alpha=0.25)

    ax.plot(
        angles,
        values_B,
        linewidth=2,
        linestyle="solid",
        label="Speaker B",
        color="#d95f02",
    )
    ax.fill(angles, values_B, color="#d95f02", alpha=0.25)

    plt.title(
        f"Gottman Profile Comparison — File {file_id}\n(% of Speaker Dialogue)",
        size=12,
        fontweight="bold",
        y=1.1,
    )
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    out_file = output_dir / f"radar_profile_file_{file_id}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()


# =====================================================================
# 4. MAIN EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "horsemen_by_file_and_speaker.csv"
    DOCS_DIR = PROJECT_ROOT / "docs"
    RADAR_DIR = DOCS_DIR / "images" / "radar_profiles"

    # Create subfolders if missing
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    RADAR_DIR.mkdir(parents=True, exist_ok=True)

    df_horsemen = pd.read_csv(DATA_PATH)

    # 1. Generate fixed heatmap
    plot_fixed_heatmap(df_horsemen, DOCS_DIR, top_n=15)

    # 2. Generate side-by-side bar reports for top high-conflict conversations
    df_horsemen["total_conflict"] = (
        df_horsemen["criticism_turns_A"]
        + df_horsemen["criticism_turns_B"]
        + df_horsemen["defensiveness_turns_A"]
        + df_horsemen["defensiveness_turns_B"]
    )
    top_conflict_files = (
        df_horsemen.sort_values(by="total_conflict", ascending=False)["file_id"]
        .head(10)
        .tolist()
    )

    for fid in top_conflict_files:
        plot_counselor_horsemen_breakdown(df_horsemen, fid, DOCS_DIR)

    # 3. BATCH GENERATE RADAR PROFILES FOR ALL FILE_IDs
    print(
        f"\n🕸️ Batch generating radar profiles for all {len(df_horsemen)} conversations..."
    )
    for _, row in df_horsemen.iterrows():
        plot_horsemen_radar_profile(row, RADAR_DIR)

    print(f"✅ All radar profiles successfully saved to: {RADAR_DIR}")
