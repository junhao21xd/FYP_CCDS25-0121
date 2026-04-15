import os
import re
import glob
import pandas as pd
import soundfile as sf  # library for fast audio processing

# --- CONFIGURATION ---
IEMOCAP_ROOT_PATH = "/path/to/SpeechCueLLM-main/IEMOCAP_full_release"
OUTPUT_CSV_PATH = "iemocap_dataset_vad_duration_full.csv"

def parse_iemocap(root_path):
    data = []
    
    # Regex pattern matches: [TIME] ID EMOTION [V, A, D]
    pattern = re.compile(r"\[.*?\]\s+(Ses\w+)\s+([a-z]{3})\s+\[(\s*-?[0-9\.]+\s*,\s*-?[0-9\.]+\s*,\s*-?[0-9\.]+\s*)\]")
    
    for session_id in range(1, 6):
        session_name = f'Session{session_id}'
        print(f"Processing {session_name}...")
        
        base_path = os.path.join(root_path, session_name)
        wav_root = os.path.join(base_path, 'sentences', 'wav')
        emo_eval_root = os.path.join(base_path, 'dialog', 'EmoEvaluation')
        trans_root = os.path.join(base_path, 'dialog', 'transcriptions')
        
        if not os.path.isdir(base_path):
            print(f"  -> Skipping {session_name} (folder not found)")
            continue

        # --- 1. Parse EmoEvaluation (Emotion + VAD) ---
        metadata_map = {} 
        
        emo_files = glob.glob(os.path.join(emo_eval_root, '*.txt'))
        for emo_file in emo_files:
            # Skip macOS '._' hidden files
            if os.path.basename(emo_file).startswith('._'):
                continue

            try:
                with open(emo_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                print(f"  -> Error reading file {os.path.basename(emo_file)}: {e}")
                continue
            
            for line in content.splitlines():
                if not line.strip().startswith('['):
                    continue
                    
                match = pattern.search(line)
                if match:
                    utt_id = match.group(1)
                    emotion = match.group(2)
                    vad_str = match.group(3)
                    
                    try:
                        val, act, dom = [float(x) for x in vad_str.split(',')]
                        metadata_map[utt_id] = {
                            'emotion': emotion,
                            'valence': val,
                            'arousal': act,
                            'dominance': dom
                        }
                    except ValueError:
                        continue

        # --- 2. Parse Transcriptions (Text) ---
        transcript_map = {}
        trans_files = glob.glob(os.path.join(trans_root, '*.txt'))
        
        for trans_file in trans_files:
            if os.path.basename(trans_file).startswith('._'):
                continue
            
            try:
                with open(trans_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
            except Exception as e:
                pass 

            for line in lines:
                parts = line.strip().split(']: ')
                if len(parts) >= 2:
                    meta = parts[0].split(' ')[0]
                    if 'Ses' in meta:
                        transcript_map[meta] = parts[1]

        # --- 3. Combine Data & Get Duration ---
        for utt_id, meta in metadata_map.items():
            
            parts = utt_id.split('_')
            video_id = "_".join(parts[:-1])
            suffix = parts[-1]
            gender_code = suffix[0] 
            
            try:
                segment_id = int(suffix[1:])
            except ValueError:
                segment_id = -1
                
            text = transcript_map.get(utt_id, "")
            
            wav_path = os.path.join(wav_root, video_id, f"{utt_id}.wav")
            
            if os.path.exists(wav_path):
                # Calculate Duration
                duration = 0.0
                try:
                    # sf.info is very fast as it only reads the header
                    audio_info = sf.info(wav_path)
                    duration = audio_info.duration
                except Exception as e:
                    print(f"  -> Warning: Could not read duration for {utt_id}: {e}")
                    duration = 0.0

                data.append({
                    "gender": gender_code,
                    "emotion": meta['emotion'],
                    "valence": meta['valence'],
                    "arousal": meta['arousal'],
                    "dominance": meta['dominance'],
                    "duration": duration,
                    "path": os.path.abspath(wav_path),
                    "text": text,
                    "id": utt_id,
                    "segment id": segment_id,
                    "video id": video_id
                })

    return pd.DataFrame(data)

if __name__ == "__main__":
    if not os.path.exists(IEMOCAP_ROOT_PATH):
        print(f"Error: Directory '{IEMOCAP_ROOT_PATH}' not found.")
    else:
        print("Starting IEMOCAP extraction with VAD & Duration...")
        df = parse_iemocap(IEMOCAP_ROOT_PATH)
        
        cols = [
            "gender", "emotion", "valence", "arousal", "dominance", 
            "duration", "path", "text", "id", "segment_id", "video_id"
        ]
        
        if not df.empty:
            df = df[cols]
            print(f"Extraction complete. Found {len(df)} samples.")
            
            # Optional: Print simple stats about duration
            print(f"Avg Duration: {df['duration'].mean():.2f}s")
            
            df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"Saved to {OUTPUT_CSV_PATH}")
        else:
            print("No data found.")
