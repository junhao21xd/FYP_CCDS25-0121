import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

# --- CONFIGURATION ---
# Change these to match your actual filenames
TRAIN_FILE = '/home/FYP/jyau005/SpeechCueLLM-main/speech_features/msp_podcast_features.csv'

# Change these to match the exact field names in your current JSON
# e.g., if your json has "val", "act", "dom", change them below.
FIELD_MAP = {
    'valence': 'EmoVal', 
    'arousal': 'EmoAct', 
    'dominance': 'EmoDom',
    'path': 'audio_filepath'  # Field containing the audio filename
}
# ---------------------

def process_file(input_path):
    print(f"Processing {input_path}...")
    
    # Extract relevant columns
    v_col = FIELD_MAP['valence']
    a_col = FIELD_MAP['arousal']
    d_col = FIELD_MAP['dominance']
        
    target_cols = [a_col, d_col, v_col]
    
    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
    except ValueError:
        # Fallback if json is line-separated (JSONL)
        df = pd.read_json(input_path, lines=True)

    
    df_train = df[df['split']=='train'].copy()
    df_ses5 = df[df['split']=='test'].copy()
    
    y = df_train['major_emotion']

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for i, (train_index, val_index) in enumerate(skf.split(df_train, y)): 
        X_train_part = df_train.iloc[train_index]
        X_val_fold = df_train.iloc[val_index].copy()
    
        output_train_csv_path = f'../msp_data_multiple/train_vad_ready_sess{i+1}.json'
        output_test_csv_path = f'../msp_data_multiple/test_vad_ready_sess{i+1}.json'

        X_train_fold = pd.concat([X_train_part, df_ses5], axis=0).copy()

        train_vals = (X_train_fold[target_cols].values - 1) / 6.0
        test_vals  = (X_val_fold[target_cols].values - 1) / 6.0
        
        train_vals = train_vals.clip(0, 1)
        test_vals = test_vals.clip(0, 1)
        
        X_train_fold['labels'] = train_vals.tolist()
        X_val_fold['labels'] = test_vals.tolist()
        
        # 5. Save new JSON with only necessary fields
        output_train_data = X_train_fold[['file', FIELD_MAP['path'], 'labels']].to_dict(orient='records')
        output_test_data = X_val_fold[['file', FIELD_MAP['path'], 'labels']].to_dict(orient='records')
        
        with open(output_train_csv_path, 'w') as f:
            json.dump(output_train_data, f, indent=2)
            
        with open(output_test_csv_path, 'w') as f:
            json.dump(output_test_data, f, indent=2)
    
    train_vals = (df_train[target_cols].values - 1) / 6.0
    test_vals  = (df_ses5[target_cols].values - 1) / 6.0

    train_vals = train_vals.clip(0, 1)
    test_vals = test_vals.clip(0, 1)

    df_train['labels'] = train_vals.tolist()
    df_ses5['labels'] = test_vals.tolist()
    
    output_train_csv_path = f'../msp_data_multiple/train_vad_ready_sess5.json'
    output_test_csv_path = f'../msp_data_multiple/test_vad_ready_sess5.json'

    # 5. Save new JSON with only necessary fields
    output_train_data = df_train[['file', FIELD_MAP['path'], 'labels']].to_dict(orient='records')
    output_test_data = df_ses5[['file', FIELD_MAP['path'], 'labels']].to_dict(orient='records')

    with open(output_train_csv_path, 'w') as f:
        json.dump(output_train_data, f, indent=2)

    with open(output_test_csv_path, 'w') as f:
        json.dump(output_test_data, f, indent=2)
 
    return 0

# --- EXECUTION ---
# 1. Process Train (and learn the min/max scaling from it)
process_file(TRAIN_FILE)

print("\nDone!")
