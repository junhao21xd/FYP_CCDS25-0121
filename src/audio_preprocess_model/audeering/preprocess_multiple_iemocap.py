import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
# Change these to match your actual filenames
TRAIN_FILE = '/path/to/SpeechCueLLM-main/speech_features/iemocap_egemaps_features_filtered_subset_sorted.csv'
output_data_path = '../iemocap_data_multiple'

# Change these to match the exact field names in your current JSON
# e.g., if your json has "val", "act", "dom", change them below.
FIELD_MAP = {
    'valence': 'valence', 
    'arousal': 'arousal', 
    'dominance': 'dominance',
    'path': 'path'  # Field containing the audio filename
}
# ---------------------

def process_file(input_path):
    print(f"Processing {input_path}...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
    except ValueError:
        # Fallback if json is line-separated (JSONL)
        df = pd.read_json(input_path, lines=True)

    for ses in range(1,6):
        df_sub_train = df[df['session']!=ses]
        df_sub_test = df[df['session']==ses]
        # 2. Extract relevant columns
        v_col = FIELD_MAP['valence']
        a_col = FIELD_MAP['arousal']
        d_col = FIELD_MAP['dominance']
        
        output_train_csv_path = f'{output_data_path}/train_vad_ready_sess{ses}.json'
        output_test_csv_path = f'{output_data_path}/test_vad_ready_sess{ses}.json'

        target_cols = [a_col, d_col, v_col]
        
        train_vals = (df_sub_train[target_cols].values - 1) / 4.0
        test_vals  = (df_sub_test[target_cols].values - 1) / 4.0
        
        train_vals = train_vals.clip(0, 1)
        test_vals = test_vals.clip(0, 1)
        
        df_sub_train['labels'] = train_vals.tolist()
        df_sub_test['labels'] = test_vals.tolist()
        
        # 5. Save new JSON with only necessary fields
        output_train_data = df_sub_train[['id', FIELD_MAP['path'], 'labels']].to_dict(orient='records')
        output_test_data = df_sub_test[['id', FIELD_MAP['path'], 'labels']].to_dict(orient='records')
        
        with open(output_train_csv_path, 'w') as f:
            json.dump(output_train_data, f, indent=2)
            
        with open(output_test_csv_path, 'w') as f:
            json.dump(output_test_data, f, indent=2)
            
    return 0

# --- EXECUTION ---
# 1. Process Train (and learn the min/max scaling from it)
process_file(TRAIN_FILE)

print("\nDone!")
