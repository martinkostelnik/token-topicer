import argparse
import json
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

from token_topicer.cross_encoder.inference import CrossEncoderInferenceModule
from token_topicer.cross_encoder.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Path to the input JSONL file for inference")
    parser.add_argument("--save-path", type=Path, default=None, help="Path to save the predictions with the best threshold")
    
    parser.add_argument("--min-threshold", type=float, default=0.1, help="Minimum threshold for predictions")
    parser.add_argument("--max-threshold", type=float, default=0.9, help="Maximum threshold for predictions")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="Step size for threshold calibration")

    return parser.parse_args()


def main():
    args = parse_args()

    with args.data.open("r") as f:
        ground_truth = [json.loads(line) for line in f]
    
    inference_module = CrossEncoderInferenceModule(model_path=args.model)
    result = inference_module.predict_jsonl(input_data_path=args.data, threshold=None)

    best_threshold = None
    best_f1 = -1.0
    best_predictions = None
    for threshold in tqdm(range(int(args.min_threshold * 100), int(args.max_threshold * 100) + 1, int(args.threshold_step * 100))):
        threshold = threshold / 100.0

        result_with_threshold = [inference_module.predict_from_probs(model_output=item, threshold=threshold) for item in result]

        f1 = evaluate(predictions=result_with_threshold, ground_truth=ground_truth)["micro_f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            if args.save_path is not None:
                best_predictions = deepcopy(result_with_threshold)
    
    print(f"Best Threshold: {best_threshold:.2f}, Best Micro F1: {best_f1:.4f}")
    if args.save_path is not None and best_predictions is not None:
        output_path = args.save_path
        with output_path.open("w") as f:
            for item in best_predictions:
                item.pop("probs")
                item.pop("offset_mapping")
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
