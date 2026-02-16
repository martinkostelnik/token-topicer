import argparse
import json
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import logging

from token_topicer.cross_encoder.inference import CrossEncoderInferenceModule
from token_topicer.cross_encoder.evaluate import evaluate


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Path to the input JSONL file for inference")
    parser.add_argument("--save-path", type=Path, default=None, help="Path to save the predictions with the best threshold (directory)")
    
    parser.add_argument("--min-threshold", type=float, default=0.1, help="Minimum threshold for predictions")
    parser.add_argument("--max-threshold", type=float, default=0.9, help="Maximum threshold for predictions")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="Step size for threshold calibration")

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    args = parse_args()

    logger.info(f"Loading ground truth data from {args.data} ...")
    with args.data.open("r") as f:
        ground_truth = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(ground_truth)} samples from ground truth data.")
    
    logger.info(f"Loading model from {args.model} ...")
    inference_module = CrossEncoderInferenceModule(model_path=args.model)
    logger.info("Model loaded successfully.")

    logger.info("Running inference on the input data ...")
    result = inference_module.predict_jsonl(input_data_path=args.data, threshold=None)
    logger.info("Inference completed.")

    logger.info("Calibrating threshold for best F1 score ...")
    best_threshold = None
    best_f1 = -1.0
    best_predictions = None
    best_evaluation_result = None
    for threshold in tqdm(range(int(args.min_threshold * 100), int(args.max_threshold * 100) + 1, int(args.threshold_step * 100))):
        threshold = threshold / 100.0

        result_with_threshold = [inference_module.predict_from_probs(model_output=item, threshold=threshold) for item in result]

        evaluation_result = evaluate(predictions=result_with_threshold, ground_truth=ground_truth)
        f1 = evaluation_result["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_evaluation_result = evaluation_result
            if args.save_path is not None:
                best_predictions = deepcopy(result_with_threshold)
    
    logger.info("Calibration completed.")
    print(f"Best Threshold: {best_threshold:.2f}, Best Macro F1: {best_f1:.4f}")

    logger.info(f"Saving predictions with the best threshold to {args.save_path} ...")
    if args.save_path is not None and best_predictions is not None:
        args.save_path.mkdir(parents=True, exist_ok=True)
        output_path_predictions = args.save_path / f"predictions_threshold_{best_threshold:.2f}.jsonl"
        output_path_evaluation = args.save_path / f"evaluation_threshold_{best_threshold:.2f}.json"
        with output_path_predictions.open("w") as f:
            for item in best_predictions:
                item.pop("probs")
                item.pop("offset_mapping")
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with output_path_evaluation.open("w") as f:
            json.dump(best_evaluation_result, f, indent=2, ensure_ascii=False)
        logger.info(f"Predictions and evaluation results saved successfully.")


if __name__ == "__main__":
    main()
