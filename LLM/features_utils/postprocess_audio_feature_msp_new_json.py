import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

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

    #features = ['Average Pitch', 'Pitch Range', 'Pitch Stability (StdDev)', 'Average Loudness', 'Overall Sound Level (dBP)', 'Loudness Range', 'Loudness Variation (StdDev)','Speaking Rate','F1 Frequency (Avg)', 'F1 Frequency (Variation)', 'F1 Amplitude (Avg)', 'F2 Frequency (Avg)', 'F2 Frequency (Variation)', 'F2 Bandwidth (Variation)', 'Voicing Rate (Segments/Sec)', 'Avg. Unvoiced Length', 'Unvoiced Length Variation', 'MFCC 1 (Spectral Shape)', 'Spectral Flux (Timbre Change)', 'Harmonic Difference (H1-H2)', 'Harmonic-Formant Diff (H1-A3)', 'Jitter (Voice Roughness)', 'Shimmer (Voice Breathiness)']
    
    features = ['Average Pitch', 'Pitch Stability (StdDev)', 'Average Loudness', 'Overall Sound Level (dBP)', 'Loudness Range', 'Loudness Variation (StdDev)', 'Avg. Loudness Decrease Slope', 'Avg. Loudness Increase Slope', 'Loudness Peaks per Second', 'Loudness 20th Percentile', 'Loudness Decrease Variability', 'Spectral Slope (500-1500 Hz)', 'Spectral Flux (Timbre Change)', 'Spectral Flux (Unvoiced Regions)', 'Spectral Flux Variation (Voiced)', 'Alpha Ratio (Spectral Balance)', 'Hammarberg Index (Voice Sharpness)', 'Speaking Rate', 'Avg. Unvoiced Length', 'Unvoiced Length Variation', 'Voiced Length Variation (StdDev)', 'MFCC 1 (Spectral Shape)', 'Harmonic-Formant Diff (H1-A3)', 'Jitter (Voice Roughness)', 'Shimmer (Voice Breathiness)', 'F1 Frequency (Avg)']
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
    for gender in ['male', 'female']:
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
    
    # Standardize features
    for feature in features:
        df[f'{feature}_standardized'] = df.apply(lambda row: 
            (row[feature] - stats.get(row['gender'], stats['overall'])[feature]['mean']) / 
            stats.get(row['gender'], stats['overall'])[feature]['std']
        if not pd.isna(row[feature]) and row[feature] != 0 else np.nan, axis=1)
    
    # Categorize original features
    for feature in features:
        df[f'{feature}_category'] = df.apply(lambda row: categorize(
            row[feature], 
            thresholds.get(row['gender'], thresholds['overall'])[feature],
            num_classes
        ), axis=1)
    
    return df

group_order = {
        "Pitch (F0)": [
            'Average Pitch',
            'Pitch Stability (StdDev)'
        ],
        "Loudness": [
            'Average Loudness', 'Overall Sound Level (dBP)',
            'Loudness Range', 'Loudness Variation (StdDev)',
            'Avg. Loudness Decrease Slope', 'Avg. Loudness Increase Slope',
            'Loudness Peaks per Second', 'Loudness 20th Percentile',
            'Loudness Decrease Variability'
        ],
        "Spectral / Formant Related": [
            'Spectral Slope (500-1500 Hz)', 'Spectral Flux (Timbre Change)',
            'Spectral Flux (Unvoiced Regions)', 'Spectral Flux Variation (Voiced)',
            'Alpha Ratio (Spectral Balance)', 'F1 Frequency (Avg)',
            'Hammarberg Index (Voice Sharpness)'
        ],
        "Pace & Timing": [
            'Speaking Rate', 'Avg. Unvoiced Length',
            'Unvoiced Length Variation', 'Voiced Length Variation (StdDev)'
        ],
        "Voice Quality": [
            'MFCC 1 (Spectral Shape)', 'Harmonic-Formant Diff (H1-A3)',
            'Jitter (Voice Roughness)', 'Shimmer (Voice Breathiness)'
        ],
    }

def generate_concise_description(row, num_classes):
    #features = ['Average Pitch', 'Pitch Range', 'Pitch Stability (StdDev)', 'Average Loudness', 'Overall Sound Level (dBP)', 'Loudness Range', 'F1 Frequency (Avg)', 'F1 Frequency (Variation)', 'F1 Amplitude (Avg)', 'F2 Frequency (Avg)', 'F2 Frequency (Variation)', 'F2 Bandwidth (Variation)', 'Voicing Rate (Segments/Sec)', 'Avg. Unvoiced Length', 'Unvoiced Length Variation', 'MFCC 1 (Spectral Shape)', 'Spectral Flux (Timbre Change)', 'Harmonic Difference (H1-H2)', 'Harmonic-Formant Diff (H1-A3)', 'Jitter (Voice Roughness)', 'Shimmer (Voice Breathiness)']
    #features = ['Average Pitch', 'Pitch Stability (StdDev)', 'Average Loudness', 'Loudness Variation (StdDev)', 'Speaking Rate']
    features = ['Average Pitch', 'Pitch Stability (StdDev)', 'Average Loudness', 'Overall Sound Level (dBP)', 'Loudness Range', 'Loudness Variation (StdDev)', 'Avg. Loudness Decrease Slope', 'Avg. Loudness Increase Slope', 'Loudness Peaks per Second', 'Loudness 20th Percentile', 'Loudness Decrease Variability', 'Spectral Slope (500-1500 Hz)', 'Spectral Flux (Timbre Change)', 'Spectral Flux (Unvoiced Regions)', 'Spectral Flux Variation (Voiced)', 'Alpha Ratio (Spectral Balance)', 'Hammarberg Index (Voice Sharpness)', 'Speaking Rate', 'Avg. Unvoiced Length', 'Unvoiced Length Variation', 'Voiced Length Variation (StdDev)', 'MFCC 1 (Spectral Shape)', 'Harmonic-Formant Diff (H1-A3)', 'Jitter (Voice Roughness)', 'Shimmer (Voice Breathiness)', 'F1 Frequency (Avg)']

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

    descriptions.append(f"Valence: {row['EmoVal']}/7\n")
    descriptions.append(f"Arousal: {row['EmoAct']}/7\n")
    descriptions.append(f"Dominance: {row['EmoDom']}/7\n")
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

    # Valence
    #if row['EmoVal_category'] != 'none':
    #    valence = category_terms[row['EmoVal_category']]
    #    descriptions.append(f"{valence} valence")
    
    # Arousal
    #if row['EmoAct_category'] != 'none':
    #    arousal = category_terms[row['EmoAct_category']]
    #    descriptions.append(f"{arousal} arousal")
    
    # Dominance
    #if row['EmoDom_category'] != 'none':
    #    dominance = category_terms[row['EmoDom_category']]
    #    descriptions.append(f"{dominance} dominance")

    # Combine all parts into a concise description
    if descriptions:
        full_description = "".join(descriptions[:-1])
    else:
        full_description = "Insufficient data to describe speech characteristics."

    return full_description

def generate_impression(row, num_classes):
    def get_level(category):
        if num_classes == 3:
            return ('medium', category)
        elif num_classes in [4, 5, 6]:
            if category in ['very_low']:
                return ('high', 'very low')
            elif category in ['low']:
                return ('medium', 'low')
            elif category in ['medium_low', 'medium', 'medium_high']:
                return ('low', 'medium')
            elif category in ['high']:
                return ('medium', 'high')
            else:  # very_high
                return ('high', 'very high')

    pitch_certainty, pitch = get_level(row['avg_pitch_category'])
    pitch_var_certainty, pitch_var = get_level(row['pitch_std_category'])
    volume_certainty, volume = get_level(row['avg_intensity_category'])
    volume_var_certainty, volume_var = get_level(row['intensity_variation_category'])
    rate_certainty, rate = get_level(row['articulation_rate_category'])

    def get_certainty_phrase(certainty):
        if certainty == 'high':
            return ""
        elif certainty == 'medium':
            return "likely "
        else:
            return "may "

    # Pitch impression
    if pitch in ['high', 'very high']:
        pitch_impression = f"{get_certainty_phrase(pitch_certainty)}uses a higher pitch"
    elif pitch in ['low', 'very low']:
        pitch_impression = f"{get_certainty_phrase(pitch_certainty)}uses a lower pitch"
    else:
        pitch_impression = "has a moderate pitch"

    if pitch_var in ['high', 'very high']:
        pitch_impression += f" with {get_certainty_phrase(pitch_var_certainty)}noticeable variation, suggesting expressiveness"
    elif pitch_var in ['low', 'very low']:
        pitch_impression += f" that {get_certainty_phrase(pitch_var_certainty)}remains steady, potentially indicating calmness or seriousness"
    else:
        pitch_impression += " with typical variation"

    # Volume impression
    if volume in ['high', 'very high']:
        volume_impression = f"{get_certainty_phrase(volume_certainty)}speaking loudly, which might indicate excitement, confidence, or urgency"
    elif volume in ['low', 'very low']:
        volume_impression = f"{get_certainty_phrase(volume_certainty)}speaking softly, possibly suggesting calmness, shyness, or caution"
    else:
        volume_impression = "using a moderate volume"

    if volume_var in ['high', 'very high']:
        volume_impression += f", with {get_certainty_phrase(volume_var_certainty)}significant volume changes"
    elif volume_var in ['low', 'very low']:
        volume_impression += f", with {get_certainty_phrase(volume_var_certainty)}little volume variation"
    else:
        volume_impression += ", with normal volume variation"

    # Speech rate impression
    if rate in ['high', 'very high']:
        rate_impression = f"{get_certainty_phrase(rate_certainty)}talking quickly, which could indicate excitement, urgency, or nervousness"
    elif rate in ['low', 'very low']:
        rate_impression = f"{get_certainty_phrase(rate_certainty)}talking slowly, possibly suggesting thoughtfulness, hesitation, or calmness"
    else:
        rate_impression = "speaking at a moderate pace"

    # Combine impressions into a single, flowing sentence
    impression = f"The target speaker {pitch_impression}, while {volume_impression}, and is {rate_impression}."
    
    return impression

def generate_feature_dict(row,num_classes):
    return 0
    

def add_conversation_history(df, window_size=3):
    """
    Creates a 'history_str' column containing the previous N turns.
    Assumes df is sorted by conversation ID and turn ID.
    """
    history_list = []
    
    # Group by Dialogue ID so we don't mix different conversations
    for conversation_id, group in df.groupby('Dialogue_ID'):
        # Iterate through turns in this conversation
        # We use a sliding window approach
        utterances = group['utterance'].tolist()
        speakers = group['Speaker'].tolist() # Optional: add speaker tags
        
        for i in range(len(group)):
            # Get previous k turns (indices i-window to i)
            start_idx = max(0, i - window_size)
            prev_turns = utterances[start_idx:i]
            prev_speakers = speakers[start_idx:i]
            
            # Format: "Speaker A: Hello \n Speaker B: Hi"
            if not prev_turns:
                hist_str = "No context available (Start of conversation)."
            else:
                lines = [f"Context (Turn {-(len(prev_turns)-j)}): {s}: {u}" 
                         for j, (s, u) in enumerate(zip(prev_speakers, prev_turns))]
                hist_str = "\n".join(lines)
            
            history_list.append(hist_str)
            
    # Add back to DF (be careful to match indices if grouping shuffled them)
    # Usually easier to apply this logic before creating the final DF
    return history_list

def add_one_line_convo(row):
    temp_content_str = 'The following conversation noted between \'### ###\' involves several speakers. The last three utterances are followed by its speech features. ### '
    temp_content_str += f"\t Speaker_{row['gender']}: {row['transcription']}"
    temp_content_str += f" ({row['Average Pitch_category']} pitch with {row['Pitch Stability (StdDev)_category']} variation)  ### \n"
    
    return temp_content_str

import pandas as pd
import json

# 1. Define your Group Order (Reference for which columns to grab)
#    This ensures we only grab the features we actually need for the prompt.
group_order = {
    "Pitch (F0)": ['Average Pitch', 'Pitch Stability (StdDev)'],
    "Loudness": [
        'Average Loudness', 'Overall Sound Level (dBP)', 'Loudness Range', 
        'Loudness Variation (StdDev)', 'Avg. Loudness Decrease Slope', 
        'Avg. Loudness Increase Slope', 'Loudness Peaks per Second', 
        'Loudness 20th Percentile', 'Loudness Decrease Variability'
    ],
    "Spectral / Formant Related": [
        'Spectral Slope (500-1500 Hz)', 'Spectral Flux (Timbre Change)',
        'Spectral Flux (Unvoiced Regions)', 'Spectral Flux Variation (Voiced)',
        'Alpha Ratio (Spectral Balance)', 'F1 Frequency (Avg)',
        'Hammarberg Index (Voice Sharpness)'
    ],
    "Pace & Timing": [
        'Speaking Rate', 'Avg. Unvoiced Length',
        'Unvoiced Length Variation', 'Voiced Length Variation (StdDev)'
    ],
    "Voice Quality": [
        'MFCC 1 (Spectral Shape)', 'Harmonic-Formant Diff (H1-A3)',
        'Jitter (Voice Roughness)', 'Shimmer (Voice Breathiness)'
    ],
}

def prepare_and_save_json(df, output_path):
    print(f"Processing {len(df)} rows...")
    final_columns = {} # Dict to map {Old_Name : New_Name}
    
    # 2a. Add Metadata columns
    final_columns['transcription'] = 'utterance'
    final_columns['major_emotion'] = 'output'
    final_columns['audio_filepath'] = 'path'
    final_columns['history_context'] = 'history_context'
    
    if 'EmoVal' in df.columns: final_columns['EmoVal'] = 'EmoVal'
    if 'EmoAct' in df.columns: final_columns['EmoAct'] = 'EmoAct'
    if 'EmoDom' in df.columns: final_columns['EmoDom'] = 'EmoDom'

    # 2b. Add Acoustic columns (and ensure they exist)
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
                df[f"{feature}_category"] = "N/A"
                final_columns[f"{feature}_category"] = f"{feature}_category"   
    
    export_df = df[list(final_columns.keys())].rename(columns=final_columns)
    
    # 3. Convert to List of Dictionaries (The "JSON" format)
    # orient='records' gives: [{col: val}, {col: val}]
    json_data = export_df.to_dict(orient='records')

    # 4. Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved to {output_path}")

def main():
    # Load the IEmocap dataset
    df = pd.read_csv('speech_features/egemaps_features_filtered.csv')
    
    # Set the number of classes (3, 4, 5, or 6)
    num_classes = 5  # Change this to 3, 4, 5, or 6 for different categorizations
    
    # Extract thresholds and stats based on the training data
    train_df = df[df['split'] == 'train']
    thresholds, stats = extract_thresholds_and_stats(train_df, num_classes)
    
    # Process the entire dataset
    processed_df = standardize_and_process_df(df, thresholds, stats, num_classes)
    
    # # for new_df with other threshold
    # new_df = pd.read_csv('speech_features/new_data2_features.csv')
    # processed_df = standardize_and_process_df(new_df, thresholds, stats, num_classes)
    
    # Generate descriptions and impressions
    # processed_df['description'] = processed_df.apply(lambda row: generate_concise_description(row, num_classes), axis=1)    
    # processed_df['impression'] = processed_df.apply(lambda row: generate_impression(row, num_classes), axis=1)
    
    processed_df['history_context'] = processed_df.apply(lambda row: add_one_line_convo(row), axis=1)

    # Create a new DataFrame with only the desired columns
    df_train = processed_df[processed_df['split']=='train']
    df_test = processed_df[processed_df['split']=='test']
    
    prepare_and_save_json(df_train, '/home/FYP/jyau005/SpeechCueLLM-main/MSP_data/train.json')
    prepare_and_save_json(df_test, '/home/FYP/jyau005/SpeechCueLLM-main/MSP_data/test.json')


if __name__ == "__main__":
    main()
