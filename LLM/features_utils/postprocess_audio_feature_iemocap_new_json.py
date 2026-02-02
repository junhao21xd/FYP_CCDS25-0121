import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import json


def plot_feature_distributions(df, features, thresholds, title, output_filename, num_classes):
    fig, axes = plt.subplots(4, 2, figsize=(20, 25))
    fig.suptitle(title, fontsize=16)
    
    for i, feature in enumerate(features):
        ax = axes[i // 2, i % 2]
        
        # Plot distribution
        sns.histplot(df[feature].replace(0, np.nan).dropna(), kde=True, ax=ax)
        
        ax.set_title(feature)
        
        # Add vertical lines for thresholds
        if num_classes == 3:
            thresholds_to_plot = ['low', 'medium_high']
        elif num_classes == 4:
            thresholds_to_plot = ['low', 'medium', 'high']
        elif num_classes == 5:
            thresholds_to_plot = ['very_low', 'low', 'medium', 'high']
        else:  # 6 classes
            thresholds_to_plot = ['very_low', 'low', 'medium_low', 'medium_high', 'high']
        
        colors = ['r', 'g', 'b', 'y', 'm']
        for threshold, color in zip(thresholds_to_plot, colors):
            if threshold in thresholds[feature]:
                ax.axvline(thresholds[feature][threshold], color=color, linestyle='--')
                ax.text(thresholds[feature][threshold], ax.get_ylim()[1], threshold,
                        ha='center', va='bottom', color=color, rotation=90)
        
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def extract_thresholds_and_stats(df, num_classes):

    features = ['valence','arousal','dominance','Average Pitch', 'Pitch Stability (StdDev)', 'Pitch Range (Dynamic Range)', 'Average Loudness', 'Overall Sound Level (dBP)', 'Loudness Variation (StdDev)', 'Avg. Loudness Increase Slope', 'Loudness Decrease Variability', 'Speaking Rate', 'Unvoiced Length Variation (Pause/Consonant Variability)', 'Jitter (Voice Roughness)', 'Shimmer (Voice Breathiness)', 'Harmonics-to-Noise Ratio (Voice Clarity)', 'Hammarberg Variation (Breathiness Stability)', 'Alpha Ratio (Spectral Balance - Voiced)', 'Alpha Ratio (Spectral Balance - Unvoiced)', 'Low Freq Spectral Slope (Voiced)', 'Low Freq Spectral Slope (Unvoiced)', 'Overall Spectral Flux Variation', 'Spectral Flux Variation (Voiced)', 'F1 Frequency (Vowel Openness)', 'F1 Amplitude (Relative to Pitch)', 'MFCC 1 (Overall Spectral Shape)', 'MFCC 1 (Voiced Spectral Shape)', 'MFCC 1 Variation (Overall)', 'MFCC 1 Variation (Voiced)', 'MFCC 2 (Voiced Spectral Energy)', 'MFCC 4 (Overall High-Freq Detail)', 'MFCC 4 (Voiced High-Freq Detail)', 'valence','arousal','dominance']
    thresholds = {}
    stats = {}
    
    # Calculate overall thresholds and stats
    overall_thresholds = {}
    overall_stats = {}
    for feature in features:
        feature_data = df[feature].replace(0, np.nan).dropna()
        if num_classes == 3:
            q1, q3 = feature_data.quantile([0.25, 0.75])
            overall_thresholds[feature] = {'low': q1, 'medium_high': q3}
        elif num_classes == 4:
            q1, q2, q3 = feature_data.quantile([0.25, 0.5, 0.75])
            overall_thresholds[feature] = {'low': q1, 'medium': q2, 'high': q3}
        elif num_classes == 5:
            q1, q2, q3, q4 = feature_data.quantile([0.1, 0.25, 0.75, 0.9])
            overall_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium': q3, 'high': q4}
        else:  # 6 classes
            q1, q2, q3, q4, q5 = feature_data.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            overall_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium_low': q3, 'medium_high': q4, 'high': q5}
        overall_stats[feature] = {'mean': feature_data.mean(), 'std': feature_data.std()}
    
    thresholds['overall'] = overall_thresholds
    stats['overall'] = overall_stats
    
    # Plot overall distribution
    # plot_feature_distributions(df, features, overall_thresholds, 
    #                            "Distribution of Audio Features (Overall)", 
    #                            'feature_distributions_overall.png',
    #                            num_classes)
    
    # Calculate gender-specific thresholds, stats, and plot
    for gender in ['M', 'F']:
        gender_df = df[df['gender'] == gender]
        gender_thresholds = {}
        gender_stats = {}
        for feature in features:
            feature_data = gender_df[feature].replace(0, np.nan).dropna()
            if len(feature_data) > 0:
                if num_classes == 3:
                    q1, q3 = feature_data.quantile([0.25, 0.75])
                    gender_thresholds[feature] = {'low': q1, 'medium_high': q3}
                elif num_classes == 4:
                    q1, q2, q3 = feature_data.quantile([0.25, 0.5, 0.75])
                    gender_thresholds[feature] = {'low': q1, 'medium': q2, 'high': q3}
                elif num_classes == 5:
                    q1, q2, q3, q4 = feature_data.quantile([0.1, 0.25, 0.75, 0.9])
                    gender_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium': q3, 'high': q4}
                else:  # 6 classes
                    q1, q2, q3, q4, q5 = feature_data.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
                    gender_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium_low': q3, 'medium_high': q4, 'high': q5}
                gender_stats[feature] = {'mean': feature_data.mean(), 'std': feature_data.std()}
            else:
                gender_thresholds[feature] = overall_thresholds[feature]
                gender_stats[feature] = overall_stats[feature]
        thresholds[gender] = gender_thresholds
        stats[gender] = gender_stats
        
        # Plot gender-specific distribution
        # plot_feature_distributions(gender_df, features, gender_thresholds, 
        #                            f"Distribution of Audio Features (Gender: {gender})", 
        #                            f'feature_distributions_gender_{gender}.png',
        #                            num_classes)
    
    return thresholds, stats

def categorize(value, thresholds, num_classes):
    if pd.isna(value) or value == 0:
        return 'none'
    if num_classes == 3:
        if value <= thresholds['low']:
            return 'low'
        elif value <= thresholds['medium_high']:
            return 'medium'
        else:
            return 'high'
    elif num_classes == 4:
        if value <= thresholds['low']:
            return 'low'
        elif value <= thresholds['medium']:
            return 'medium_low'
        elif value <= thresholds['high']:
            return 'medium_high'
        else:
            return 'high'
    elif num_classes == 5:
        if value <= thresholds['very_low']:
            return 'Very low'
        elif value <= thresholds['low']:
            return 'Low'
        elif value <= thresholds['medium']:
            return 'Medium'
        elif value <= thresholds['high']:
            return 'High'
        else:
            return 'Very high'
    else:  # 6 classes
        if value <= thresholds['very_low']:
            return 'very_low'
        elif value <= thresholds['low']:
            return 'low'
        elif value <= thresholds['medium_low']:
            return 'medium_low'
        elif value <= thresholds['medium_high']:
            return 'medium_high'
        elif value <= thresholds['high']:
            return 'high'
        else:
            return 'very_high'

def standardize_and_process_df(df, thresholds, stats, num_classes):
    features = list(thresholds['overall'].keys())
    VAD_FEATURES = {'valence', 'arousal', 'dominance'} 
    # Standardize features
    #for feature in features: 
        #df[f'{feature}_standardized'] = df.apply(lambda row: 
        #    (row[feature] - stats.get(row['gender'], stats['overall'])[feature]['mean']) / 
        #    stats.get(row['gender'], stats['overall'])[feature]['std']
        #if not pd.isna(row[feature]) and row[feature] != 0 else np.nan, axis=1)
    
    # Categorize original features
    #for feature in features:
    #    df[f'{feature}_category'] = df.apply(lambda row: categorize(
    #        row[feature], 
    #        thresholds.get(row['gender'], thresholds['overall'])[feature],
    #        num_classes
    #    ), axis=1)
    
    for feature in features:
        is_vad = feature in VAD_FEATURES    
        # --- 1. Standardization ---
        df[f'{feature}_standardized'] = df.apply(lambda row: _standardize_row(
            row, feature, stats, is_vad
        ), axis=1)

        # --- 2. Categorization ---
        df[f'{feature}_category'] = df.apply(lambda row: _categorize_row(
            row, feature, thresholds, num_classes, is_vad
        ), axis=1)
    return df

def _standardize_row(row, feature, stats, is_vad):
    # If VAD, always use 'overall'. If Acoustic, try gender, fallback to 'overall'
    if is_vad:
        target_stats = stats['overall'][feature]
    else:
        target_stats = stats.get(row['gender'], stats['overall'])[feature]
    
    val = row[feature]
    if pd.isna(val) or val == 0:
        return np.nan
        
    return (val - target_stats['mean']) / target_stats['std']

def _categorize_row(row, feature, thresholds, num_classes, is_vad):
    # Same logic: VAD = Global, Acoustic = Gendered
    if is_vad:
        target_thresh = thresholds['overall'][feature]
    else:
        target_thresh = thresholds.get(row['gender'], thresholds['overall'])[feature]
        
    return categorize(row[feature], target_thresh, num_classes)

group_order = {
    "Pitch (Intonation)": [
        'Average Pitch',
        'Pitch Stability (StdDev)',
        'Pitch Range (Dynamic Range)',
    ],

    "Loudness (Energy)": [
        'Average Loudness',
        'Overall Sound Level (dBP)',
        'Loudness Variation (StdDev)',
        'Avg. Loudness Increase Slope',
        'Loudness Decrease Variability',
    ],

    "Rhythm & Speed": [
        'Speaking Rate',
        'Unvoiced Length Variation (Pause/Consonant Variability)',
    ],

    "Voice Quality (Roughness/Breathiness)": [
        'Jitter (Voice Roughness)',
        'Shimmer (Voice Breathiness)',
        'Harmonics-to-Noise Ratio (Voice Clarity)',
        'Hammarberg Variation (Breathiness Stability)',
    ],

    "Spectral Balance (Timbre & Tone)": [
        'Alpha Ratio (Spectral Balance - Voiced)',
        'Alpha Ratio (Spectral Balance - Unvoiced)',
        'Low Freq Spectral Slope (Voiced)',
        'Low Freq Spectral Slope (Unvoiced)',
        'Overall Spectral Flux Variation',
        'Spectral Flux Variation (Voiced)',
    ],

    "Formants (Vowel Articulation)": [
        'F1 Frequency (Vowel Openness)',
        'F1 Amplitude (Relative to Pitch)',
    ],

    "MFCCs (Abstract Spectral Shape)": [
        'MFCC 1 (Overall Spectral Shape)',
        'MFCC 1 (Voiced Spectral Shape)',
        'MFCC 1 Variation (Overall)',
        'MFCC 1 Variation (Voiced)',
        'MFCC 2 (Voiced Spectral Energy)',
        'MFCC 4 (Overall High-Freq Detail)',
        'MFCC 4 (Voiced High-Freq Detail)',
    ],
}


def generate_concise_description(row, num_classes):

    descriptions = ["Target speech characteristics:\n"]

    # Mapping of categories to descriptive terms
    if num_classes == 3:
        category_terms = {
            'low': 'low', 'medium': 'moderate', 'high': 'high'
        }
    elif num_classes == 4:
        category_terms = {
            'low': 'low', 'medium_low': 'moderately low', 
            'medium_high': 'moderately high', 'high': 'high'
        }
    elif num_classes == 5:
        category_terms = {
            'very_low': 'very low', 'low': 'low', 'medium': 'moderate', 
            'high': 'high', 'very_high': 'very high'
        }
    else:  # 6 classes
        category_terms = {
            'very_low': 'extremely low', 'low': 'very low', 'medium_low': 'low',
            'medium_high': 'moderate', 'high': 'high', 'very_high': 'extremely high'
        }

    descriptions.append(f"Valence: {row['valence_category']}\n")
    descriptions.append(f"Arousal: {row['arousal_category']}\n")
    descriptions.append(f"Dominance: {row['dominance_category']}\n")
    descriptions.append(f"\n")
    #for feature in features:  # keeps your given order
    #    category_col = f"{feature}_category"
        
        # make sure the column exists and isn't 'none'
    #    if category_col in row and row[category_col] != 'none':
    #        category_value = row[category_col]
    #        descriptions.append(f"{feature}: {category_value}\n")

    for key in group_order.keys():
        feature_list = group_order[key]
        descriptions.append(f"--- {key} ---\n")
        for feature in feature_list:
            category_col = f"{feature}_category"

            # make sure the column exists and isn't 'none'
            if category_col in row and row[category_col] != 'none':
                category_value = row[category_col]
                descriptions.append(f"{feature}: {category_value}\n")
        descriptions.append("\n")

    # Combine all parts into a concise description
    if descriptions:
        full_description = "".join(descriptions[:-1])
    else:
        full_description = "Insufficient data to describe speech characteristics."

    return full_description    

# def add_conversation_history(df, window_size=13):
#     """
#     Creates a 'history_str' containing the previous N turns (including current).
#     Logic: 
#       - Row 1: [1]
#       - Row 3: [1, 2, 3]
#       - Row 4: [2, 3, 4]
#     """
#     # 1. Sort to ensure temporal order
#     df = df.sort_values(['video_id', 'segment_id'])
    
#     # Initialize column to avoid SettingWithCopy warnings
#     df['history_str'] = ""
    
#     # 2. Iterate through each distinct conversation
#     for conversation_id, group in df.groupby('video_id', sort=False):
#         texts = group['text'].tolist()
#         indices = group.index.tolist()

#         # Generate Speaker Labels (A, B, A, B...)
#         speakers = ['A' if k % 2 == 0 else 'B' for k in range(len(texts))]
        
#         for i, row_idx in enumerate(indices):
#             # --- SLIDING WINDOW LOGIC ---
#             # We want the slice [start : end]
#             # End is i+1 because Python slices are exclusive at the end
#             # Start is max(0, end - window_size)
            
#             end_pos = i + 1
#             start_pos = max(0, end_pos - window_size)
            
#             # Slice both texts and speakers
#             window_texts = texts[start_pos:end_pos]
#             window_speakers = speakers[start_pos:end_pos]

#             # Build the string
#             lines = [
#                 f"Speaker {s}: {u}"
#                 for s, u in zip(window_speakers, window_texts)
#             ]

#             # Assign back to the original dataframe
#             # .at is faster than .loc for scalar assignment
#             df.at[row_idx, 'history_str'] = "\n".join(lines)

#     return df

# def add_conversation_history_and_filter(df, window_size=3, max_samples=12):
#     """
#     1. Creates 'history_str' for the LAST N turns of each conversation.
#     2. Filters the dataframe to keep ONLY those last N turns.
#     """
#     df = df.sort_values(['video_id', 'segment_id'])
#     df['history_str'] = ""
    
#     # We will collect the indices of the rows we want to KEEP
#     indices_to_keep = []

#     for conversation_id, group in df.groupby('video_id', sort=False):
#         texts = group['text'].tolist()
#         indices = group.index.tolist()
        
#         # Calculate where the "Last 12" starts
#         total_turns = len(indices)
#         start_index = max(0, total_turns - max_samples)
        
#         # Identify the subset of rows we want to keep (The last 12)
#         keep_indices = indices[start_index:]
#         indices_to_keep.extend(keep_indices)

#         # Generate History for the rows we are keeping
#         # Note: We still access 'texts' from before start_index for context!
#         speakers = ['A' if k % 2 == 0 else 'B' for k in range(total_turns)]
        
#         # We loop only through the range we intend to keep (e.g., 88 to 100)
#         for i in range(start_index, total_turns):
#             row_idx = indices[i]
            
#             # Sliding Window Logic
#             # We look back 'window_size' steps from current position i
#             # This correctly grabs text from "deleted" rows if needed for context
#             context_end = i + 1
#             context_start = max(0, context_end - window_size)
            
#             window_texts = texts[context_start:context_end]
#             window_speakers = speakers[context_start:context_end]

#             lines = [
#                 f"Speaker {s}: {u}"
#                 for s, u in zip(window_speakers, window_texts)
#             ]
#             df.at[row_idx, 'history_str'] = "\n".join(lines)

#     # Final Step: Filter the DataFrame to only the rows we processed
#     df_filtered = df.loc[indices_to_keep].copy()
    
#     print(f"Original samples: {len(df)}")
#     print(f"Filtered samples (Max {max_samples}/convo): {len(df_filtered)}")
    
#     return df_filtered

def add_conversation_history(df, window_size=8):
    """
    Creates a 'history_str' for IEMOCAP using strict logic from data_process.py.
    
    Logic:
      - Window: Current utterance + previous 'window_size' utterances.
      - Format: Tab-separated (\t), matching the original script.
      - Speakers: Uses df['gender'] (e.g., 'M'/'F'). 
        (Note: Original script mapped M->0, F->1. If you prefer that, 
         you can map the column before running this.)
    """
    # 1. Sort to ensure temporal order
    # Assuming 'segment_id' or 'turn_index' exists to order utterances within a dialogue
    df = df.sort_values(by=['video_id', 'Order_Index']) 
    
    # Initialize column
    df['history_context'] = ""
    
    # 2. Iterate through each distinct conversation
    for conversation_id, group in df.groupby('video_id', sort=False):
        texts = group['text'].tolist()
        # Use the explicit gender column as requested
        speakers = group['gender'].tolist()
        indices = group.index.tolist()
        pitches = group['Average Pitch_category'].tolist()
        variations = group['Pitch Stability (StdDev)_category'].tolist()
 
        for i, row_idx in enumerate(indices):
            # --- STRICT ORIGINAL SCRIPT LOGIC ---
            # In data_process.py: index_w = max(conv_turn - window, 0)
            # It iterates from index_w up to conv_turn (inclusive).
            
            # This creates a slice of length (window_size + 1)
            # i.e., The current turn + the 12 previous turns.
            start_pos = max(0, i - window_size)
            end_pos = i + 1
            
            # Slice the lists
            w_texts = texts[start_pos:end_pos]
            w_speakers = speakers[start_pos:end_pos]
            w_pitches = pitches[start_pos:end_pos]
            w_variations = variations[start_pos:end_pos]

            lines = []
            window_len = len(w_texts)

            # Iterate through the current window slice
            for k in range(window_len):
                s = w_speakers[k]
                u = w_texts[k]
                p = w_pitches[k]
                v = w_variations[k]

                # Base string
                utterance_str = f'Speaker_{s}:"{u}"'

                # --- NEW LOGIC: Add features only to the last 3 items ---
                # "k" is the index within this specific window (0 to window_len-1).
                # If window has 5 items, indices are 0,1,2,3,4. We want 2,3,4.
                if k >= window_len - 3:
                    utterance_str += f' ({p} pitch with {v} variation)'
                
                lines.append(utterance_str)

            # Join with tabs (\t) to match the original script's "temp_content_str" style
            # The original script adds a tab before every speaker.
            df.at[row_idx, 'history_context'] = "\t " + "\t ".join(lines)

    return df

def add_one_line_convo(row):
    temp_content_str = 'The following conversation noted between \'### ###\' involves several speakers. The last three texts are followed by its speech features. ### '
    temp_content_str += f"\t Speaker_{row['gender']}: {row['text']}"
    temp_content_str += f" ({row['Average Pitch_category']} pitch with {row['Pitch Stability (StdDev)_category']} variation)  ### \n"
    
    return temp_content_str


def prepare_and_save_json(df, output_path):
    print(f"Processing {len(df)} rows...")
    final_columns = {} # Dict to map {Old_Name : New_Name}
    
    # 2a. Add Metadata columns
    final_columns['text'] = 'utterance'
    final_columns['emotion'] = 'output'
    final_columns['path'] = 'path'
    final_columns['history_context'] = 'history_context'
    
    if 'valence_category' in df.columns: final_columns['valence_category'] = 'valence'
    if 'arousal_category' in df.columns: final_columns['arousal_category'] = 'arousal'
    if 'dominance_category' in df.columns: final_columns['dominance_category'] = 'dominance'

    # Acoustic features
    for group, features in group_order.items():
        for feature in features:
            # Check if your DF has "Average Pitch" OR "Average Pitch_category"
            if f"{feature}_category" in df.columns:
                final_columns[f"{feature}_category"] = f"{feature}_category"
            elif feature in df.columns:
                final_columns[feature] = f"{feature}_category" # Rename it!
            else:
                # If missing, create a placeholder so code doesn't crash
                print(f"Warning: Missing {feature}, filling with 'N/A'")
                df[f"{feature}_category"] = "Unknown"
                final_columns[f"{feature}_category"] = f"{feature}_category"   
    
    iemocap_to_target = {
        'hap': 'happy',
        'happy': 'happy',

        'sad': 'sad',

        'neu': 'neutral',
        'neutral': 'neutral',

        'ang': 'angry',
        'anger': 'angry',

        'exc': 'excited',
        'excited': 'excited',

        'fru': 'frustrated',
        'frustrated': 'frustrated',
    }
    df['emotion'] = df['emotion'].map(iemocap_to_target) 
    export_df = df[list(final_columns.keys())].rename(columns=final_columns)

    json_data = export_df.to_dict(orient='records')
    
    # 4. Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved to {output_path}")

def main():
    # Load the IEmocap dataset
    df = pd.read_csv('speech_features/iemocap_egemaps_features_filtered_subset_sorted.csv')
    
    # Set the number of classes (3, 4, 5, or 6)
    num_classes = 5  # Change this to 3, 4, 5, or 6 for different categorizations
    
    # Extract thresholds and stats based on the training data
    train_df = df[df['split'] == 'train']
    thresholds, stats = extract_thresholds_and_stats(train_df, num_classes)
    
    # Process the entire dataset
    processed_df = standardize_and_process_df(df, thresholds, stats, num_classes)
    
    #processed_df['history_context'] = processed_df.apply(lambda row: add_conversation_history(row), axis=1)
    processed_df = add_conversation_history(processed_df)

    # Create a new DataFrame with only the desired columns
    df_train = processed_df[processed_df['split']=='train']
    df_test = processed_df[processed_df['split']=='test']
    
    prepare_and_save_json(df_train, '/home/FYP/jyau005/SpeechCueLLM-main/IEMOCAP_data_norm_VAD_grouped/train.json')
    prepare_and_save_json(df_test, '/home/FYP/jyau005/SpeechCueLLM-main/IEMOCAP_data_norm_VAD_grouped/test.json')


if __name__ == "__main__":
    main()
