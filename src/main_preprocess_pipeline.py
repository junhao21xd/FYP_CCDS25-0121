import argparse
import os
from audio_preprocess_model import run_asr

def main():
    parser = argparse.ArgumentParser(description="Master Audio Pipeline")
    parser.add_argument("--dataset", type=str.upper, required=True)
    parser.add_argument("--model_id", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--calc_wer", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--input_json", type=str, help="input audio file to run asr on")
    parser.add_argument("--output_json", type=str, help="output audio file that stores asr transcription")
    
    args = parser.parse_args()

    DATA_ROOT_DIR = "../data"
    if not args.input_json:
        args.input_json = f"{DATA_ROOT_DIR}/{args.dataset}_data/test.json"
    
    if not args.output_json:
        args.output_json = f"{DATA_ROOT_DIR}/{args.dataset}_data_ASR/test.json"

    # Check 1: Does the input file actually exist?
    if not os.path.isfile(args.input_json):
        parser.error(
            f"\nFATAL ERROR: Input file not found at:\n{args.input_json}\n"
            f"Please check your dataset name or provide the exact path using --input_json."
        )

    # Check 2: Does the output directory exist?
    output_dir = os.path.dirname(args.output_json)
    
    if not os.path.exists(output_dir):
        print(f"[*] Notice: Output directory '{output_dir}' not found. Creating it automatically...")
        os.makedirs(output_dir, exist_ok=True)
    
    run_asr(
        input_json=args.input_json,
        output_json=args.output_json,
        model_id=args.model_id,
        calc_wer=args.calc_wer,
        normalize=args.normalize
    )

if __name__ == "__main__":
    main()