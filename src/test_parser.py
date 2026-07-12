from pathlib import Path  # Ensure this is imported at the very top of the file

import pylangacq


def test_callhome_parsing():
    # Modern pathlib syntax instead of os.path.join
    data_dir = Path("data") / "callhome"

    # Modern recursive glob tracking
    cha_files = list(data_dir.rglob("*.cha"))

    if not cha_files:
        print(f"❌ Error: No .cha files found in '{data_dir}'.")
        print(
            "Please double-check that your CallHome files are placed inside that folder."
        )
        return

    print(f"✅ Found {len(cha_files)} .cha files.")
    sample_file = cha_files[0]
    print(f"Testing parser on sample file: {sample_file}\n")

    try:
        # 2. Read the sample file using pylangacq
        chat_data = pylangacq.read_chat(sample_file)

        # 3. Pull the first 5 utterances to verify the text and timestamps
        utterances = chat_data.utterances()
        print("--- First 5 Conversational Turns ---")

        for i, turn in enumerate(utterances[:5]):
            speaker = turn.participant

            # Extract the actual words from the token objects
            clean_text = " ".join([token.word for token in turn.tokens])

            # The correct property name is time_marks (not time_markers)
            timestamps = turn.time_marks

            print(f"Turn {i + 1} | Speaker: {speaker} | Time: {timestamps}")
            print(f"Text: \"{clean_text}\"")
            print("-" * 40)

        print("\n🎉 Success! The dataset parsed beautifully.")

    except Exception as e:
        print(f"❌ An error occurred while parsing the file: {e}")


if __name__ == "__main__":
    test_callhome_parsing()
