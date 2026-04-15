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

# --- CONFIGURATION ---
EXTRACTOR_PATH = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

#MODEL_PATH = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim" # Pretrained model path
MODEL_PATH = "/path/to/audeer/wav2vec2_vad_iemocap_final_fp16"  # Path where you saved the trained model

TEST_FILE = "../iemocap_data/test_vad_ready.json"    # Path to your processed test json

output_csv = "iemocap_predictions.csv"
BATCH_SIZE = 8
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
            "path": path # Keep track of filename
        }

def collate_fn(batch):
    # Custom collate to handle padding
    processor = Wav2Vec2Processor.from_pretrained(EXTRACTOR_PATH)
    input_values = [x["input_values"] for x in batch]
    labels = [x["labels"] for x in batch]
    paths = [x["path"] for x in batch]
    
    padded_inputs = processor.feature_extractor.pad(
        {"input_values": input_values},
        padding=True,
        return_tensors="pt"
    )
    
    return {
        "input_values": padded_inputs.input_values,
        "attention_mask": padded_inputs.attention_mask,
        "labels": torch.stack(labels),
        "paths": paths
    }

# 4. MAIN EVALUATION LOOP
def evaluate():
    print(f"Loading model from {MODEL_PATH}...")
    processor = Wav2Vec2Processor.from_pretrained(EXTRACTOR_PATH)
    model = EmotionModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    print(f"Loading test data from {TEST_FILE}...")
    dataset = EvalDataset(TEST_FILE, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Storage for results
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

            # Forward pass
            logits = model(input_values, attention_mask=attention_mask)
            preds = logits.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)
            all_paths.extend(paths)

    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # 5. CALCULATE METRICS
    # Model Output Order: [Arousal, Dominance, Valence]
    dims = ["Arousal", "Dominance", "Valence"]
    results = {}

    print("\n" + "="*40)
    print(" EVALUATION RESULTS ")
    print("="*40)
    print(f"{'Dimension':<12} | {'CCC':<8} | {'PCC':<8} | {'MSE':<8}")
    print("-" * 44)

    for i, dim in enumerate(dims):
        pred = all_preds[:, i]
        true = all_labels[:, i]

        ccc = concordance_correlation_coefficient(true, pred)
        pcc, _ = pearsonr(true, pred)
        mse = np.mean((true - pred)**2)

        results[dim] = {"CCC": ccc, "PCC": pcc, "MSE": mse}
        print(f"{dim:<12} | {ccc:.4f}   | {pcc:.4f}   | {mse:.4f}")

    print("="*40)
    
    # Average CCC (often used as the single main metric)
    avg_ccc = np.mean([results[d]['CCC'] for d in dims])
    print(f"Average CCC: {avg_ccc:.4f}")

    # 6. SAVE PREDICTIONS TO CSV
    df = pd.DataFrame({
        "path": all_paths,
        "pred_arousal": all_preds[:, 0],
        "true_arousal": all_labels[:, 0],
        "pred_dominance": all_preds[:, 1],
        "true_dominance": all_labels[:, 1],
        "pred_valence": all_preds[:, 2],
        "true_valence": all_labels[:, 2],
    })
    df.to_csv(output_csv, index=False)
    print("\nDetailed predictions saved to 'predictions.csv'")

if __name__ == "__main__":
    evaluate()
