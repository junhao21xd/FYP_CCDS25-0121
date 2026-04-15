from datasets import load_dataset, Dataset, ClassLabel, concatenate_datasets
import os
import soundfile as sf
import pandas as pd
import re

# Load full dataset
ds = load_dataset("AbstractTTS/PODCAST", split="train")
# ds = ds.shuffle(seed=42)
# Directory to save wavs
wav_dir = "/path/to/MSP_dataset/wav_outputs"
os.makedirs(wav_dir, exist_ok=True)

create_wav_file = False
    
def compute_duration(example):
    # example['audio'] is a dict: {'array': [...], 'sampling_rate': 16000}
    audio_array = example['audio']['array']
    sampling_rate = example['audio']['sampling_rate']
    duration = len(audio_array) / sampling_rate
    return {"duration": duration}

# Apply to entire dataset
ds = ds.map(compute_duration, batched=True, batch_size=1000)

ds = ds.rename_column(
    "major_emotion", "major_emotion_str"
)
ds = ds.add_column(
    "major_emotion", ds["major_emotion_str"]
)

unique_emotions = sorted(set(ds["major_emotion_str"]))
class_label = ClassLabel(names=unique_emotions)

ds = ds.cast_column(
    "major_emotion", class_label
)

split_dataset = ds.train_test_split(
    test_size=0.2,
    stratify_by_column="major_emotion",
    seed=42
)

train_ds = split_dataset['train'].add_column("split", ["train"] * len(split_dataset['train']))
test_ds = split_dataset['test'].add_column("split", ["test"] * len(split_dataset['test']))

ds = concatenate_datasets([train_ds, test_ds])

def add_path(example):
    audio = example['audio']
    array = audio['array']
    sampling_rate = audio['sampling_rate']
    filename = example['file']
    filepath = os.path.join(wav_dir, filename)
    sf.write(filepath, array, sampling_rate)
    return {"path": filepath}

if create_wav_file:
    ds = ds.map(add_path)

ds = ds.remove_columns("audio")

df = ds.to_pandas()

# Function to clean text while maintaining sentence flow
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Fix formatting: "noise] actual_text [" → keep only the "actual_text"
    match = re.search(r'\]\s*(.*?)\s*\[', text)
    if match:
        text = match.group(1)

    # 2. Replace [crosstalk ...] and [inaudible ...] with placeholders
    text = re.sub(r'\[crosstalk[^\]]*?\]', '[crosstalk]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[inaudible[^\]]*?\]', '[inaudible]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[laughter[^\]]*?\]', '[laughter]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[foreign language[^\]]*?\]', '[inaudible]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[in audible[^\]]*?\]', '[inaudible]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[german[^\]]*?\]', '[inaudible]', text, flags=re.IGNORECASE)

    # 4. Replace patterns like [Tom 00:02:24] → Tom
    text = re.sub(
        r'\[([^\[\]]*?)\s+\d\d:\d\d(?::\d\d)?(?:\.\d+)?\]',  # e.g., [Tom 00:02:24]
        r'\1',
        text
    )
    
    text = re.sub(r'\[\s*\d{1,2}(:\d{2}){1,2}(\.\d+)?\s*\]', '', text)

    # 5. Remove brackets around remaining content, preserving inner text
    text = re.sub(r'\[([^\[\]]+)\]', r'\1', text)

    # 6. Optional: Remove any unmatched brackets (e.g., stray [ or ])
    text = text.replace('[', '').replace(']', '')

    # Remove specific label tags (case insensitive)
    return text.strip()

# Apply the cleaning function and drop empty rows

df = df.dropna(subset=['transcription'])

df['transcription'] = df['transcription'].apply(lambda x: clean_text(x))
df = df[df['transcription'] != ""]

df.to_csv("msp_podcast_complete.csv",index=False)
