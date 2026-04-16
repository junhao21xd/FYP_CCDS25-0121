import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm 
import librosa

def preprocess_function(examples, feature_extractor):
        audio_arrays = []
        
        for path in examples["path"]: 
            array, _ = librosa.load(path, sr=16000)
            audio_arrays.append(array)
            
        inputs = feature_extractor(
            audio_arrays, 
            sampling_rate=16000, 
            max_length=16000 * 5, 
            truncation=True,
            padding=True,
            return_attention_mask=True
        )
        return inputs

def predict_genders(df, extractor_path, model_path):
    hf_test = Dataset.from_pandas(df)
    
    print("Loading model and feature extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(extractor_path)
    model = AutoModelForAudioClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()     
    print("Processing audio files...")
    encoded_test = hf_test.map(preprocess_function, batched=True, batch_size=8, fn_kwargs={"feature_extractor": feature_extractor})

    encoded_test.set_format(type="torch", columns=["input_values", "attention_mask"])

    dataloader = DataLoader(encoded_test, batch_size=8)

    # 7. Run the Inference Loop
    print("Running inference...")
    all_preds = []
    all_probs = []

    with torch.no_grad(): 
        for batch in tqdm(dataloader):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_values, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            class_1_probs = probs[:, 1].cpu().numpy()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(class_1_probs)

    results_df = df.copy()

    id2label = model.config.id2label

    results_df['gender_pred'] = [id2label[p] for p in all_preds]
    results_df['gender_probability'] = all_probs

    return results_df

def run_gender_classifier_inference(data_path, extractor_path, model_path, output_csv):
    df = pd.read_csv(data_path)
    if "audio_filepath" in df.columns:
        df = df.rename(columns={"audio_filepath": "path"})
    
    results_df = predict_genders(df, extractor_path, model_path)
    
    final_csv_df = results_df[['id', 'gender_pred', 'gender_probability']]
    final_csv_df.to_csv(output_csv, index=False)
    
    print(f"\nPredictions successfully saved to {output_csv}")
    
def run_gender_classifier_evaluation(data_path, extractor_path, model_path, output_csv):
    print(f"Evaluating dataset in {data_path}")
    df = pd.read_csv(data_path)
    if "audio_filepath" in df.columns:
        df = df.rename(columns={"audio_filepath": "path"})
    df = df[df['split'] == 'test'].copy()
    
    results_df = predict_genders(df, extractor_path, model_path)
    
    true_labels = results_df['gender']
    print("\nWav2Vec 2.0 Evaluation Results:")
    print("-" * 30)
    print(f"Accuracy: {accuracy_score(true_labels, results_df['gender_pred']):.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, results_df['gender_pred']))
    
    final_csv_df = results_df[['id', 'gender_pred', 'gender_probability']]
    final_csv_df.to_csv(output_csv, index=False)
    
    print(f"\nPredictions successfully saved to {output_csv}")
    