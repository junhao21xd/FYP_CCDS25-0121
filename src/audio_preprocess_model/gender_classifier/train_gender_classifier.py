import pandas as pd
import numpy as np
from datasets import Dataset, Audio, DatasetDict, ClassLabel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate
import librosa


DATASET_CONFIGS = {
    "iemocap": {
        "eval_strategy":"epoch",
        "save_strategy":"epoch",
        "logging_steps":10,
        "learning_rate":3e-5,            
        "per_device_train_batch_size":8, 
        "per_device_eval_batch_size":8,
        "num_train_epochs":5,
        "load_best_model_at_end":True,
        "metric_for_best_model":"accuracy",
        "save_total_limit":1,
        "gradient_checkpointing":True
    },
    
    "msp": {
        "eval_strategy":"epoch",
        "save_strategy":"epoch",
        "logging_steps":10,
        "learning_rate":3e-5,            
        "per_device_train_batch_size":8, 
        "per_device_eval_batch_size":8,
        "num_train_epochs":5,
        "load_best_model_at_end":True,
        "metric_for_best_model":"accuracy",
        "save_total_limit":1,
        "gradient_checkpointing":True,
        "max_grad_norm":1.0,       
        "warmup_ratio":0.1,     
        "weight_decay":0.01,
    },
    
    "default": {
        "eval_strategy":"epoch",
        "save_strategy":"epoch",
        "logging_steps":10,
        "learning_rate":1e-5,            
        "per_device_train_batch_size":8, 
        "per_device_eval_batch_size":8,
        "num_train_epochs":5,
        "load_best_model_at_end":True,
        "metric_for_best_model":"accuracy",
        "save_total_limit":1,
        "gradient_checkpointing":True,
        "max_grad_norm":1.0,       
        "warmup_ratio":0.1,     
        "weight_decay":0.01,
    }
}

def df_to_hf_dataset(data_path):
    df = pd.read_csv(data_path)
    
    # rename due to col name mismatch in datasets
    if "audio_filepath" in df.columns:
        df = df.rename(columns={"audio_filepath": "path"})
    
    df = df[['id','gender','split','path']]

    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    hf_dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    return hf_dataset

def preprocess_function(examples, feature_extractor):
    audio_arrays = []
    
    for path in examples["path"]: 
        array, _ = librosa.load(path, sr=16000)
        trimmed_array, _ = librosa.effects.trim(array, top_db=30)
        audio_arrays.append(trimmed_array)
        
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        max_length=16000 * 5, 
        truncation=True,
        padding=True
    )
    return inputs


def train_gender_classifier(dataset, data_path, feature_extractor_path, model_checkpoint):
    finetuned_model_path = f"./wav2vec2-gender-best-model_{dataset}"
    hf_dataset = df_to_hf_dataset(data_path)
    
    labels = sorted(hf_dataset["train"].unique("gender"))
    class_label = ClassLabel(names=labels)
    
    hf_dataset = hf_dataset.cast_column("gender", class_label)
    hf_dataset = hf_dataset.rename_column("gender", "label")
    
    label2id = {name: i for i, name in enumerate(class_label.names)}
    id2label = {i: name for i, name in enumerate(class_label.names)}

    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

    # Apply preprocessing to train and test sets
    print("Processing audio files...")
    encoded_dataset = hf_dataset.map(preprocess_function, batched=True, fn_kwargs={"feature_extractor": feature_extractor})

    # 7. Setup Evaluation Metric
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)

    # 8. Load Pre-trained Model with a new Classification Head
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label
    )

    current_config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS["DEFAULT"])
    
    # 9. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./wav2vec2-gender-classifier_{dataset}",
        **current_config
    )

    # 10. Initialize Trainer and Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    print("Starting wav2vec 2.0 fine-tuning...")
    trainer.train()
    trainer.save_model(finetuned_model_path)
    return finetuned_model_path
