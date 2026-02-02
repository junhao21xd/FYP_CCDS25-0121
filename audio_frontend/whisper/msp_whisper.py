import torch
import json
import os
import soundfile as sf
import jiwer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# --- CONFIGURATION ---
INPUT_JSON = "/path/to/SpeechCueLLM-main/MSP_data/test.json"          # Your input file
OUTPUT_JSON = "/path/to/SpeechCueLLM-main/MSP_data_ASR/test.json" # The final output file
MODEL_ID = "openai/whisper-large-v3-turbo"

# --- 1. SETUP MODEL ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading {MODEL_ID} on {device}...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, 
    torch_dtype=dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True, 
    attn_implementation="eager"
)
model.to(device)
processor = AutoProcessor.from_pretrained(MODEL_ID)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=dtype,
    device=device,
)

normalizer = BasicTextNormalizer()

# --- 2. LOAD DATA ---
print(f"Reading from {INPUT_JSON}...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Processing {len(data)} items...")
total_errors = 0
total_ref_words = 0
# --- 3. PROCESSING LOOP ---
# We loop through the data and modify the dictionary items in-place
for i, item in enumerate(data):
    audio_path = item["path"]
    ground_truth = item.get("utterance", "")

    # Extract ID for logging (optional, keeps your console clean)
    file_id = os.path.splitext(os.path.basename(audio_path))[0]
    
    try:
        # A. Load Audio (FFmpeg Bypass)
        audio_data, samplerate = sf.read(audio_path)
        
        # B. Transcribe
        result = pipe(
            {"raw": audio_data, "sampling_rate": samplerate},
            return_timestamps=True, generate_kwargs={"language": "english"}
        )
#        result = pipe(
#            audio_data
#        )
 
        # C. Normalize
        # hyp_norm = normalizer(result["text"])
        # ref_norm = normalizer(ground_truth)

        hyp_norm = result["text"]
        ref_norm = ground_truth
        
        # D. Calculate WER
        # Handle edge cases (empty strings)
        #if len(ref_norm) > 0:
        #    sample_wer = jiwer.wer(ref_norm, hyp_norm)
        #else:
        #    sample_wer = 1.0 if len(hyp_norm) > 0 else 0.0
        
        # E. Update the Item
        # We add new keys to the existing dictionary
        output = jiwer.process_words(ref_norm, hyp_norm)
        
        errors = output.substitutions + output.deletions + output.insertions
        ref_len = len(ref_norm.split())
        total_errors += errors
        total_ref_words += ref_len
        item["hypothesis"] = hyp_norm
        item["wer"] = round(output.wer, 4)
        
        # Optional: Print progress every 10 items
        if (i + 1) % 10 == 0:
            #print(f"[{i+1}/{len(data)}] Processed. WER: {sample_wer:.2f}")
            if total_ref_words > 0:
                current_global_wer = total_errors / total_ref_words
                print(f"[{i+1}/{len(data)}] Current Global WER: {current_global_wer:.2%}")
            else:
                print(f"[{i+1}/{len(data)}] Processed (No ref words yet)")

    except Exception as e:
        print(f"❌ Error on {file_id}: {e}")
        item["hypothesis"] = "ERROR"
        item["wer"] = 1.0

# --- 4. SAVE EVERYTHING IN ONE GO ---
print(f"Saving final results to {OUTPUT_JSON}...")
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

# Print Final Summary
if total_ref_words > 0:
    final_wer = total_errors / total_ref_words
    print("="*30)
    print(f"✅ FINISHED.")
    print(f"Total Words Processed: {total_ref_words}")
    print(f"Final Global WER: {final_wer:.2%}")
    print("="*30)

#print("✅ Done.")
