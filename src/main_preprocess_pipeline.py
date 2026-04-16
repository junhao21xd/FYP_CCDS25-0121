import argparse
import os
from audio_preprocess_model import run_asr, run_gender_classifier_evaluation, run_gender_classifier_inference, train_gender_classifier

def main():
    parser = argparse.ArgumentParser(description="Master Audio Pipeline")
    parser.add_argument("--dataset", type=str.upper, required=True)
    parser.add_argument("--asr_model_id", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--calc_wer", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--asr_input_json", type=str, help="input audio file to run asr on")
    parser.add_argument("--asr_output_json", type=str, help="output audio file that stores asr transcription")
    
    parser.add_argument("--feature_extractor_path", default="facebook/wav2vec2-base")
    parser.add_argument("--gender_classifier_path", default="facebook/wav2vec2-base")
    parser.add_argument("--train_gender_classifier", default=False)
    parser.add_argument("--eval_gender_classifier", default=False)
    parser.add_argument("--gender_input_csv", type=str, help="input audio file to classify gender")
    parser.add_argument("--gender_output_csv", type=str, help="output audio file that stores gender prediction and probability")
    
    args = parser.parse_args()

    DATA_ROOT_DIR = "../data"
    if not args.asr_input_json:
        args.asr_input_json = f"{DATA_ROOT_DIR}/{args.dataset}_data/test.json"
    
    if not args.asr_output_json:
        args.asr_output_json = f"{DATA_ROOT_DIR}/{args.dataset}_data_ASR/test.json"

    # Check 1: Does the input file actually exist?
    if not os.path.isfile(args.asr_input_json):
        parser.error(
            f"\nFATAL ERROR: Input file not found at:\n{args.asr_input_json}\n"
            f"Please check your dataset name or provide the exact path using --input_json."
        )

    # Check 2: Does the output directory exist?
    output_dir = os.path.dirname(args.asr_output_json)
    
    if not os.path.exists(output_dir):
        print(f"[*] Notice: Output directory '{output_dir}' not found. Creating it automatically...")
        os.makedirs(output_dir, exist_ok=True)
    
    run_asr(
        input_json=args.asr_input_json,
        output_json=args.asr_output_json,
        model_id=args.asr_model_id,
        calc_wer=args.calc_wer,
        normalize=args.normalize
    )

    if args.train_gender_classifier:
        finetuned_model_path = train_gender_classifier(args.dataset, args.gender_input_csv, args.feature_extractor_path, args.gender_classifier_path)
    
    else:
        finetuned_model_path = args.gender_classifier_path
    
    if args.eval_gender_classifier:
        run_gender_classifier_evaluation(args.gender_input_csv, args.feature_extractor_path, finetuned_model_path, args.gender_output_csv)

    else:
        run_gender_classifier_inference(args.gender_input_csv, args.feature_extractor_path, finetuned_model_path, args.gender_output_csv)

if __name__ == "__main__":
    main()