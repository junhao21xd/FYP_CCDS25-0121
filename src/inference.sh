#!/bin/bash

# Exit immediately if any command fails
set -e

# ==============================================================================
# 1. USER CONFIGURATION
# Modify the variables below to change the behavior of the Python scripts.
# ==============================================================================

# --- Virtual Environment Paths ---
VENV_AUDIO="../venv_audio/bin/python3"
VENV_LLM="../venv_llm/bin/python3"


    parser.add_argument("--train_gender_classifier", default=False)
    parser.add_argument("--eval_gender_classifier", default=False)
    parser.add_argument("--gender_input_csv", type=str, help="input audio file to classify gender")
    parser.add_argument("--gender_output_csv", type=str, help="output audio file that stores gender prediction and probability")

# --- Args ---
DATASET="iemocap"

# --- Args for run_asr.py ---
MODEL_ID="openai/whisper-large-v3-turbo"

# --- Args for run_gender_classifer.py ---
GENDER_CLASSIFIER_PATH="./wav2vec2-gender-best-model_${DATASET}"
GENDER_INPUT_CSV="../data/speech_features/${DATASET}_egemaps_features_filtered_subset_sorted.csv"
GENDER_OUTPUT_CSV="${DATASET}_wav2vec_predictions.csv"

# for msp gender classifier
# data_path = "../data/speech_features/egemaps_features_filtered.csv"

# ==============================================================================
# 2. ENVIRONMENT SETUP
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# 3. SCRIPT EXECUTION
# ==============================================================================
"$VENV_AUDIO" main_preprocess_pipeline.py \
    --dataset "$DATASET" \
    --asr_model_id "$MODEL_ID" \
    --gender_classifier_path "$GENDER_CLASSIFIER_PATH" \
    --gender_input_csv "$GENDER_INPUT_CSV" \
    --gender_output_csv "$GENDER_OUTPUT_CSV"

# "$VENV_LLM" another_script.py 
