import pandas as pd
import numpy as np
from datasets import Dataset, Audio, DatasetDict
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate
import librosa
# 1. Convert your Pandas DataFrame into a Hugging Face Dataset
# (Assuming 'df' is your existing dataframe with 'path' and 'gender' columns)

df = pd.read_csv('/home/FYP/jyau005/SpeechCueLLM-main/speech_features/iemocap_egemaps_features_filtered_subset_sorted.csv')
df = df[['id','gender','split','path']]

train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

hf_dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# 2. Create Label Mappings
labels = hf_dataset["train"].unique("gender")
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

# Map string labels ('M', 'F') to integers (0, 1)
def label_to_id(example):
    example["label"] = int(label2id[example["gender"]])
    return example

hf_dataset = hf_dataset.map(label_to_id, remove_columns=["gender"])

# 3. The Magic Step: Automatic Audio Loading and Resampling
# Wav2vec 2.0 strictly requires 16kHz audio. This command tells the dataset 
# to automatically read the file at 'path' and resample it on the fly when called!
#hf_dataset = hf_dataset.cast_column("path", Audio(sampling_rate=16000))

# Rename 'path' to 'audio' as expected by the Hugging Face Trainer
#hf_dataset = hf_dataset.rename_column("path", "audio")

# 4. Split the dataset
# NOTE: Replace this random split with your exact Session 1-4 vs Session 5 split logic for IEMOCAP!


# 5. Load the Feature Extractor
model_checkpoint = "facebook/wav2vec2-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

# 6. Preprocess
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
        padding=True
    )
    return inputs

# Apply preprocessing to train and test sets
print("Processing audio files...")
encoded_dataset = hf_dataset.map(preprocess_function, batched=True)

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

# 9. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-gender-classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=3e-5,            # A very small learning rate is crucial so we don't overwrite the pre-trained weights!
    per_device_train_batch_size=8, # If you get a CUDA Out of Memory error, drop this to 4 or 2
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    gradient_checkpointing=True
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
trainer.save_model("./wav2vec2-gender-best-model")

