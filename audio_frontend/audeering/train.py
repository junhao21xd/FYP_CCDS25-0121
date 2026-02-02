import numpy as np
import torch
import torch.nn as nn
import librosa
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
import json
# ---------------------------------------------------------
# 1. LOSS FUNCTION (Concordance Correlation Coefficient)
# ---------------------------------------------------------
class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gold):
        """
        Computes 1 - CCC. 
        Pred and Gold shape: [batch_size, 3] (Arousal, Dominance, Valence)
        """
        loss = 0
        # Calculate CCC for each dimension (A, D, V) separately and average
        for i in range(pred.shape[1]):
            p = pred[:, i]
            g = gold[:, i]
            
            p_mean = torch.mean(p)
            g_mean = torch.mean(g)
            
            p_var = torch.var(p, unbiased=False)
            g_var = torch.var(g, unbiased=False)
            
            covariance = torch.mean((p - p_mean) * (g - g_mean))
            
            numerator = 2 * covariance
            denominator = p_var + g_var + (p_mean - g_mean)**2
            
            ccc = numerator / (denominator + 1e-8)
            loss += (1.0 - ccc)
            
        return loss / pred.shape[1]

# ---------------------------------------------------------
# 2. MODEL DEFINITION (Modified for Training)
# ---------------------------------------------------------
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
        
        # Define Loss
        self.loss_fct = CCCLoss()

        # FREEZE FEATURE EXTRACTOR (Crucial for Method B)
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
            self,
            input_values,
            attention_mask=None,
            labels=None,
            **kwargs
    ):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        
        # Mean pooling
        if attention_mask is None:
             hidden_states = torch.mean(hidden_states, dim=1)
        else:
            # Proper masking for variable length audio
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            hidden_states = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
            #padding_mask = padding_mask.unsqueeze(-1).type_as(hidden_states)
            #hidden_states = hidden_states * padding_mask
            #hidden_states = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).clamp(min=1e-6)

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return (loss, logits) if loss is not None else logits

# ---------------------------------------------------------
# 3. DATASET AND COLLATOR
# ---------------------------------------------------------
class VADDataset(Dataset):
    def __init__(self, data_list, processor, sampling_rate=16000):
        """
        data_list: List of dicts [{'path': '/path/to/audio.wav', 'labels': [A, D, V]}]
        """
        self.data_list = data_list
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Load audio (Replace this with your actual loading logic)
        # Using librosa to ensure 16kHz
        audio, _ = librosa.load(item['path'], sr=self.sampling_rate)
        
        # --- DUMMY AUDIO FOR DEMONSTRATION ---
        #audio = np.random.uniform(-1, 1, 16000 * 3) # 3 seconds noise
        # -------------------------------------

        # Process audio
        inputs = self.processor(audio, sampling_rate=self.sampling_rate, return_tensors="pt")
        
        return {
            "input_values": inputs.input_values[0],
            "labels": torch.tensor(item['labels'], dtype=torch.float32)
        }

@dataclass
class DataCollatorWithPadding:
    processor: Wav2Vec2Processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_values = [{"input_values": feature["input_values"]} for feature in features]
        labels = [feature["labels"] for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_values,
            return_tensors="pt",
            padding=True
        )
        
        #batch["labels"] = torch.stack(labels)
        batch["labels"] = torch.stack(labels).float()
        return batch

# ---------------------------------------------------------
# 4. TRAINING SETUP
# ---------------------------------------------------------

# Load Pretrained Components
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)

with open('../data/train_vad_ready.json') as f:
    train_data = json.load(f)

with open('../data/test_vad_ready.json') as f:
    eval_data = json.load(f)

# --- PREPARE DUMMY DATA ---
# Replace this with your CSV loading logic
# Labels should be: [Arousal, Dominance, Valence]
#train_data = [{'path': 'file1.wav', 'labels': [0.5, 0.5, 0.5]} for _ in range(20)]
#eval_data = [{'path': 'file2.wav', 'labels': [0.6, 0.4, 0.7]} for _ in range(5)]

train_dataset = VADDataset(train_data, processor)
eval_dataset = VADDataset(eval_data, processor)
data_collator = DataCollatorWithPadding(processor=processor)

# Metrics
def compute_metrics(p: EvalPrediction):
    #preds = p.predictions
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    labels = p.label_ids
    # Simple MSE for logging (Loss handles CCC)
    mse = ((preds - labels)**2).mean().item()
    return {"mse": mse}

# Training Arguments
training_args = TrainingArguments(
    output_dir="../wav2vec2_vad_finetuned_fp16",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    num_train_epochs=5,
    fp16=True if torch.cuda.is_available() else False,
    #fp16=False,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    learning_rate=1e-4, # Higher LR for the head
    save_total_limit=1,
    remove_unused_columns=False,
    label_names=["labels"],
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start Training
print("Starting training...")
trainer.train()

# Save final model
trainer.save_model("../wav2vec2_vad_iemocap_final_fp16")
#processor.save_pretrained("../wav2vec2_vad_iemocap_final")
