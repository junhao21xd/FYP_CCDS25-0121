import opensmile
import pandas as pd
import glob


# Initialize eGeMAPS (extended version)
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPS,  # eGeMAPSv02
    feature_level=opensmile.FeatureLevel.Functionals
)

# Folder with your audio files
audio_folder = "/path/to/IEMOCAP_full_release/Session*/sentences/wav/*/*.wav"

# Store results
all_features = []

# Loop through each audio file
for file in glob.glob(audio_folder):
    if file.endswith('.wav') and not file.startswith('._'):
        features = smile.process_file(file)
        features['file'] = file  # Keep track of filename
        all_features.append(features)

# Combine all into a single DataFrame
df = pd.concat(all_features)
df.reset_index(drop=True, inplace=True)

# Save to CSV
df.to_csv("../speech_features/iemocap_egemaps_features.csv", index=False)

print("Features extracted and saved to egemaps_features.csv")



