import os
import re
import pandas as pd
import glob

def create_order_index(iemocap_root_path):
    """
    Parses IEMOCAP transcriptions to create a chronologically sorted index.
    """
    data = []
    transcription_dirs = glob.glob(iemocap_root_path)

    for tdir in transcription_dirs: 
    # Walk through the dataset structure
    # Expected structure: SessionX/dialog/transcriptions/
        for root, dirs, files in os.walk(tdir):
            for file in files:
                if file.endswith(".txt") and not file.startswith("."):
                    
                    # The filename (minus .txt) is usually the Dialogue_ID
                    dialogue_id = os.path.splitext(file)[0]
                    file_path = os.path.join(root, file)
                    
                    # We collect all turns for this specific dialogue here
                    dialogue_turns = []
                    
                    with open(file_path, 'r') as f:
                        for line in f:
                            # Regex to find: UtteranceID [Start-End]:
                            # Matches: Ses01F_impro01_F000 [6.2900-8.2350]:
                            match = re.match(r"(Ses\w+)\s+\[(\d+\.\d+)-(\d+\.\d+)\]:", line)
                            
                            if match:
                                utt_id = match.group(1)
                                start_time = float(match.group(2))
                                
                                dialogue_turns.append({
                                    "video_id": dialogue_id,
                                    "id": utt_id,
                                    "Start_Time": start_time
                                })
                    
                    # SORTING LOGIC:
                    # 1. Sort this specific dialogue by Start_Time
                    dialogue_turns.sort(key=lambda x: x["Start_Time"])
                    
                    # 2. Add the reset index (0, 1, 2...)
                    for idx, turn in enumerate(dialogue_turns):
                        turn["Order_Index"] = idx
                        data.append(turn)

    return pd.DataFrame(data)

# --- USAGE ---

# 1. Generate the mapping
# Replace with your actual IEMOCAP path
iemocap_path = "/path/to/IEMOCAP_full_release/Session*/dialog/transcriptions" 
df_order = create_order_index(iemocap_path)

# 2. Load your existing CSV
# Assuming your CSV has a column 'Utterance_ID' to match on
my_existing_csv = pd.read_csv("/path/to/speech_features/iemocap_egemaps_features_filtered_subset.csv") 

# 3. Merge the new 'Order_Index' into your data
# We merge on Utterance_ID to ensure the index matches the specific wav file
df_final = pd.merge(my_existing_csv, df_order[['id', 'Order_Index']], 
                    on='id', 
                    how='left')

# 4. Final Sort
# Sort by Dialogue ID first, then by the new Order Index
df_final = df_final.sort_values(by=['video_id', 'Order_Index'])

print(df_final.head())
df_final.to_csv('/path/to/speech_features/iemocap_egemaps_features_filtered_subset_sorted.csv',index=False)
