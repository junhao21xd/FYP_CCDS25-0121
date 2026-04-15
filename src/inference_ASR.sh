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

# --- Args for whisper.py ---
DATASET="IEMOCAP"
MODEL_ID="openai/whisper-large-v3-turbo"

# --- Args for another_script.py (Example) ---
# LEARNING_RATE="0.001"
# BATCH_SIZE="32"

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
    --model_id "$MODEL_ID"

# "$VENV_LLM" another_script.py 
