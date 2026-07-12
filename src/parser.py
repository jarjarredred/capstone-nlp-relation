# Script using pylangacq to clean .cha files

from pathlib import Path  # Replace 'import os' and 'import glob'
import re

import pandas as pd
import pylangacq


def clean_text_string(tokens):
    """
    Applies Text Cleaning & Data Formatting (Techniques 3 & 7):
    Strips transcription artifacts, lowercases, and removes extra spaces.
    """
    words = []
    for token in tokens:
        word = token.word
        # Remove transcription noise codes like &uh, &laughter, or bracketed text
        if word.startswith('&') or word.startswith('[') or word.endswith(']'):
            continue
        # Remove structural transcription punctuation markers like <, >, /, *
        word = re.sub(r'[<>\/\*]', '', word)

        # Lowercase text (Technique 3)
        word = word.lower().strip()

        if word:
            words.append(word)

    return " ".join(words)


def build_clean_tabular_dataset():
    # Modern pathlib syntax instead of os.path.join
    data_dir = Path("data") / "callhome"
    output_dir = Path("data") / "processed"

    # Modern directory creation
    output_dir.mkdir(parents=True, exist_ok=True)

    # Modern globbing using rglob (recursive glob)
    cha_files = list(data_dir.rglob("*.cha"))

    if not cha_files:
        print(f"❌ Error: No .cha files found in '{data_dir}'.")
        return

    print(f"🧹 Starting data cleaning pipeline on {len(cha_files)} files...")

    master_records = []

    for file_path in cha_files:
        # file_path is a Path object now, so .stem gets the filename without extension
        file_id = file_path.stem

        try:
            chat_data = pylangacq.read_chat(file_path)
            utterances = chat_data.utterances()

            for turn_idx, turn in enumerate(utterances):
                speaker = turn.participant

                # 1. Text Cleaning & Data Formatting (Techniques 3 & 7)
                clean_text = clean_text_string(turn.tokens)

                # Handling Missing Data (Technique 1): Drop rows with no actual spoken text
                if not clean_text.strip():
                    continue

                # Extract structural variables for time-series features
                timestamps = turn.time_marks  # (start_ms, end_ms)
                start_ms = timestamps[0] if timestamps else None
                end_ms = timestamps[1] if timestamps else None

                # Append raw structured record
                master_records.append(
                    {
                        "file_id": file_id,
                        "turn_index": turn_idx,
                        "speaker_raw": speaker,
                        "text": clean_text,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                    }
                )
        except Exception as e:
            print(f"⚠️ Skipping file {file_id} due to parsing error: {e}")

    # Convert to Pandas DataFrame
    df = pd.DataFrame(master_records)

    # 2. Removing Duplicates (Technique 2)
    # Ensure every conversation sequence is unique per file
    df = df.drop_duplicates(subset=["file_id", "turn_index"])

    # 3. Handling Missing Data / Formatting Imputation (Technique 1 & 3)
    # Chronologically forward-fill missing timestamps within each conversation file
    df["start_ms"] = df.groupby("file_id")["start_ms"].ffill()
    df["end_ms"] = df.groupby("file_id")["end_ms"].ffill()

    # 4. Encoding Categorical Data (Technique 6)
    # Convert 'A' and 'B' into binary 0 and 1 tags
    df["speaker_encoded"] = df["speaker_raw"].map({"A": 0, "B": 1})
    # Drop rows where speaker didn't match standard A/B tags
    df = df.dropna(subset=["speaker_encoded"])
    df["speaker_encoded"] = df["speaker_encoded"].astype(int)

    # 5. Basic Normalization Metric Preparation (Technique 5)
    # Add an absolute word count per turn for normalization thresholds later
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))

    # Modern slash operator joining
    output_path = output_dir / "callhome_tabular_clean.csv"
    df.to_csv(output_path, index=False)

    print("\n🎉 Pipeline Complete!")
    print(f"📊 Total Structured Rows Cleaned: {df.shape[0]}")
    print(f"💾 File saved successfully to: {output_path}")


if __name__ == "__main__":
    build_clean_tabular_dataset()
