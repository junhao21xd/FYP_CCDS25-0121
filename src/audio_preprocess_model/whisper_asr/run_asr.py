import argparse
import torch
import json
import os
import soundfile as sf
import jiwer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

'''
Generate transcription of audio in (input_json) with ASR model (model_id).
The output is stored in (output_json).
Calculate Word Error Rate (WER) or normalised WER if requested.
'''
    
def run_asr(input_json, output_json, model_id="openai/whisper-large-v3-turbo", calc_wer=False, normalize_wer=False):    
    # --- 1. SETUP MODEL ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading {model_id} on {device}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True, 
        attn_implementation="eager"
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

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
    print(f"Reading from {input_json}...")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} items...")
    total_errors = 0
    total_ref_words = 0
    
    for i, item in enumerate(data):
        audio_path = item.get("path") or item.get("audio_filepath")
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

            # C. Normalize
            if normalize_wer:
                hyp = normalizer(result["text"]).strip()
                ref = normalizer(ground_truth).strip()

            else:
                hyp = result["text"]
                ref = ground_truth
            
            if calc_wer and ref !="":            
                output = jiwer.process_words(ref, hyp)
                errors = output.substitutions + output.deletions + output.insertions
                ref_len = len(ref.split())
                total_errors += errors
                total_ref_words += ref_len
                item["wer"] = round(output.wer, 4)
            
            else:
                item["wer"] = None
                
            item["hypothesis"] = result["text"]
            
            # Optional: Print progress every 10 items
            # if (i + 1) % 10 == 0:
            #     if total_ref_words > 0:
            #         current_global_wer = total_errors / total_ref_words
            #         print(f"[{i+1}/{len(data)}] Current Global WER: {current_global_wer:.2%}")
            #     else:
            #         print(f"[{i+1}/{len(data)}] Processed (No ref words yet)")

        except Exception as e:
            print(f"❌ Error on {file_id}: {e}")
            item["hypothesis"] = "ERROR"
            item["wer"] = None

    # --- 4. SAVE EVERYTHING IN ONE GO ---
    print(f"Saving final results to {output_json}...")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    # Print Final Summary
    if calc_wer:
        final_wer = total_errors / total_ref_words
        print("="*30)
        print(f"FINISHED.")
        print(f"Total Words Processed: {total_ref_words}")
        print(f"Final Global WER: {final_wer:.2%}")
        print("="*30)

