import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
# Change these to match your actual filenames
TRAIN_FILE = '../../data/MSP_data/train.json'
TEST_FILE = '../../data/MSP_data/test.json'
output_data_path = '../../data/audeering/msp_data'
# Change these to match the exact field names in your current JSON
# e.g., if your json has "val", "act", "dom", change them below.
FIELD_MAP = {
    'valence': 'valence', 
    'arousal': 'arousal', 
    'dominance': 'dominance',
    'path': 'path'  # Field containing the audio filename
}
# ---------------------

def process_file(input_path, output_path, scaler=None):
    print(f"Processing {input_path}...")
    
    # 1. Load Data
    try:
        df = pd.read_json(input_path)
    except ValueError:
        # Fallback if json is line-separated (JSONL)
        df = pd.read_json(input_path, lines=True)

    # 2. Extract relevant columns
    v_col = FIELD_MAP['valence']
    a_col = FIELD_MAP['arousal']
    d_col = FIELD_MAP['dominance']
    
    # 3. Fit Scaler (Normalize to 0-1)
    # We fit the scaler ONLY on the train set, then apply it to test
    # to prevent data leakage.
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit on the 3 columns
        scaler.fit(df[[a_col, d_col, v_col]]) 
        
    # Transform the values
    scaled_values = scaler.transform(df[[a_col, d_col, v_col]])
    
    df['labels'] = scaled_values.tolist()
    
    # 5. Save new JSON with only necessary fields
    output_data = df[[FIELD_MAP['path'], 'labels']].to_dict(orient='records')
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Saved {len(output_data)} samples to {output_path}")
    return scaler

# --- EXECUTION ---
# 1. Process Train (and learn the min/max scaling from it)
scaler = process_file(TRAIN_FILE, f'{output_data_path}/train_vad_ready.json', scaler=None)

# 2. Process Test (using the scaling logic from Train)
process_file(TEST_FILE, f'{output_data_path}/test_vad_ready.json', scaler=scaler)

print("\nDone! Use 'train_vad_ready.json' and 'test_vad_ready.json' in your training script.")
