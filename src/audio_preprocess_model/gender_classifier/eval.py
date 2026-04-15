import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm # Provides a nice progress bar
import librosa
# 1. Load your Session 5 test dataframe
# Assuming 'test_df' is already loaded and contains 'id', 'path', and 'gender'
# test_df = df_combine[df_combine['split'] == 'test'].copy()

df = pd.read_csv('/home/FYP/jyau005/SpeechCueLLM-main/speech_features/iemocap_egemaps_features_filtered_subset_sorted.csv')
df = df[['id','gender','split','path']]

train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']


# 2. Define the path to your fine-tuned model directory
# The Trainer saved the best model here at the end of training
extractor_path = "facebook/wav2vec2-base"
model_path = "./wav2vec2-gender-best-model" 

print("Loading model and feature extractor...")
feature_extractor = AutoFeatureExtractor.from_pretrained(extractor_path)
model = AutoModelForAudioClassification.from_pretrained(model_path)

# 3. Move model to GPU (if available) and set to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Crucial: disables dropout layers for stable inference

# 4. Prepare the Hugging Face Dataset (Keep the 'id' column safe!)
hf_test = Dataset.from_pandas(test_df)
#hf_test = hf_test.cast_column("path", Audio(sampling_rate=16000))

# 5. Preprocessing Function
def preprocess_function(examples):
    audio_arrays = []
    
    # examples["path"] is now just a list of string file paths
    for path in examples["path"]: 
        # librosa.load reads the file and forces it to 16kHz instantly
        # It returns a tuple: (numpy_array, sample_rate)
        array, _ = librosa.load(path, sr=16000)
        audio_arrays.append(array)
        
    # Pass the manually loaded arrays directly into the feature extractor
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        max_length=16000 * 5, 
        truncation=True,
        padding=True,
        return_attention_mask=True
    )
    return inputs

print("Processing audio files...")
encoded_test = hf_test.map(preprocess_function, batched=True, batch_size=8)

# Convert outputs to PyTorch tensors so the model can read them
encoded_test.set_format(type="torch", columns=["input_values", "attention_mask"])

# 6. Create a PyTorch DataLoader for batching
dataloader = DataLoader(encoded_test, batch_size=8)

# 7. Run the Inference Loop
print("Running inference...")
all_preds = []
all_probs = []

with torch.no_grad(): # Disable gradient calculation to save memory and speed up inference
    for batch in tqdm(dataloader):
        # Move inputs to the GPU
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Forward pass through wav2vec 2.0
        outputs = model(input_values, attention_mask=attention_mask)
        
        # Extract raw, unnormalized scores (logits)
        logits = outputs.logits
        
        # Convert logits to percentages (probabilities) using Softmax
        probs = F.softmax(logits, dim=-1)
        
        # Get the predicted class index (0 or 1) by finding the highest probability
        preds = torch.argmax(probs, dim=-1)
        
        # Store the probability of Class 1 
        # (This is your rich "soft label" for the emotion classifier)
        class_1_probs = probs[:, 1].cpu().numpy()
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(class_1_probs)



# 8. Reconstruct Results DataFrame
results_df = test_df.copy()

# The model config automatically saved your label mapping during training!
id2label = model.config.id2label

# Map the integer predictions (e.g., 0, 1) back to your string labels ('F', 'M')
results_df['gender_pred'] = [id2label[p] for p in all_preds]
results_df['gender_probability'] = all_probs

# 9. Evaluate Final Accuracy
true_labels = results_df['gender']
print("\nWav2Vec 2.0 Evaluation Results:")
print("-" * 30)
print(f"Accuracy: {accuracy_score(true_labels, results_df['gender_pred']):.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, results_df['gender_pred']))

# 10. Save to CSV for your merging step
final_csv_df = results_df[['id', 'gender_pred', 'gender_probability']]
final_csv_df.to_csv('iemocap_wav2vec_predictions.csv', index=False)
print("\nPredictions successfully saved to iemocap_wav2vec_predictions.csv")
