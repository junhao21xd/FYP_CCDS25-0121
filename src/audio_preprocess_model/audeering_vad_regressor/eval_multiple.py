import torch
import torch.nn as nn
import numpy as np
import json
import pandas as pd
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Model
from tqdm import tqdm
from scipy.stats import pearsonr
import glob
import os
# --- CONFIGURATION ---
dataset = 'iemocap'

EXTRACTOR_PATH = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

BATCH_SIZE = 8                       # Adjust based on your GPU VRAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------

# 1. MODEL DEFINITION (Must match training script exactly)
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        
        # Masked Mean Pooling
        if attention_mask is None:
             hidden_states = torch.mean(hidden_states, dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            hidden_states = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(hidden_states)
        return logits

# 2. METRICS
def concordance_correlation_coefficient(y_true, y_pred):
    """Calculates CCC for a single dimension."""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    numerator = 2 * correlation * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    
    return numerator / denominator

# 3. DATASET
class EvalDataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.sr = 16000

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        id = item['id']
        path = item['path']
        labels = item['labels'] # [Arousal, Dominance, Valence]

        # LOAD AUDIO
        # Warning: Ensure your paths in JSON are absolute or correct relative to this script
        audio, _ = librosa.load(path, sr=self.sr)
        
        # Process
        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt")
        
        return {
            "input_values": inputs.input_values[0],
            "labels": torch.tensor(labels, dtype=torch.float32),
            "path": path, # Keep track of filename
            "id": id
        }

def collate_fn(batch):
    # Custom collate to handle padding
    processor = Wav2Vec2Processor.from_pretrained(EXTRACTOR_PATH)
    input_values = [x["input_values"] for x in batch]
    labels = [x["labels"] for x in batch]
    paths = [x["path"] for x in batch]
    ids = [x["id"] for x in batch]
    
    padded_inputs = processor.feature_extractor.pad(
        {"input_values": input_values},
        padding=True,
        return_tensors="pt"
    )
    
    return {
        "input_values": padded_inputs.input_values,
        "attention_mask": padded_inputs.attention_mask,
        "labels": torch.stack(labels),
        "paths": paths,
        "ids": ids
    }

# 4. MAIN EVALUATION LOOP
def evaluate():
    all_sessions_dfs = []
    for ses in range(1,6):
        MODEL_PATH = f"../wav2vec2_vad_{dataset}_finetuned_sess{ses}" 
        TEST_FILE = f"../{dataset}_data_multiple/test_vad_ready_sess{ses}.json"    # Path to your processed test json
        
        checkpoints = glob.glob(os.path.join(MODEL_PATH, "checkpoint-*"))

        print(f"Loading model from {MODEL_PATH}...")
        processor = Wav2Vec2Processor.from_pretrained(EXTRACTOR_PATH)
        model = EmotionModel.from_pretrained(checkpoints[0]).to(DEVICE)
        model.eval()

        print(f"Loading test data from {TEST_FILE}...")
        dataset = EvalDataset(TEST_FILE, processor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        # Storage for results
        all_ids = []
        all_preds = []
        all_labels = []
        all_paths = []

        print("Running Inference...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_values = batch["input_values"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].numpy() # Keep labels on CPU
                paths = batch["paths"]
                ids = batch["ids"]
                # Forward pass
                logits = model(input_values, attention_mask=attention_mask)
                preds = logits.cpu().numpy()

                all_preds.append(preds)
                all_labels.append(labels)
                all_paths.extend(paths)
                all_ids.extend(ids)
                
        # Concatenate all batches
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # 5. CALCULATE METRICS
        # Model Output Order: [Arousal, Dominance, Valence]
        dims = ["Arousal", "Dominance", "Valence"]
        results = {}

        output_str = []
        output_str.append("="*40)
        output_str.append(f" EVALUATION RESULTS - SESSION {ses} ")
        output_str.append("="*40)
        output_str.append(f"{'Dimension':<12} | {'CCC':<8} | {'PCC':<8} | {'MSE':<8}")
        output_str.append("-" * 44)

        for i, dim in enumerate(dims):
            pred = all_preds[:, i]
            true = all_labels[:, i]

            ccc = concordance_correlation_coefficient(true, pred)
            pcc, _ = pearsonr(true, pred)
            mse = np.mean((true - pred)**2)

            results[dim] = {"CCC": ccc, "PCC": pcc, "MSE": mse}
            output_str.append(f"{dim:<12} | {ccc:.4f}   | {pcc:.4f}   | {mse:.4f}")
        output_str.append("="*40)
        
        # Average CCC (often used as the single main metric)
        avg_ccc = np.mean([results[d]['CCC'] for d in dims])
        output_str.append(f"Average CCC: {avg_ccc:.4f}")

        final_report = "\n".join(output_str)
        print(final_report)
        
        report_path = f"../{dataset}_result/score_summary_sess{ses}.txt"
        with open(report_path, "w") as f:
            f.write(final_report)
        print(f"Saved score summary to {report_path}")
    
        # 6. SAVE PREDICTIONS TO CSV
        df = pd.DataFrame({
            "session": ses,
            "id": all_ids,
            "path": all_paths,
            "pred_arousal": all_preds[:, 0],
            "true_arousal": all_labels[:, 0],
            "pred_dominance": all_preds[:, 1],
            "true_dominance": all_labels[:, 1],
            "pred_valence": all_preds[:, 2],
            "true_valence": all_labels[:, 2],
        })
        all_sessions_dfs.append(df)
    
    full_df = pd.concat(all_sessions_dfs, ignore_index=True)    
    full_df.to_csv("../{dataset}_result/{dataset}_predictions_full.csv", index=False)
    print("\npredictions saved")

if __name__ == "__main__":
    evaluate()
