import argparse
import json
from pathlib import Path
import logging

from token_topicer.utils import split_chunk_into_words

from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
import numpy as np


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Cross Encoder model for topic tagging.")
    parser.add_argument("--pred", type=Path, required=True, help="Path to the predictions file.")
    parser.add_argument("--gt", type=Path, required=True, help="Path to the ground truth data file.")

    parser.add_argument("--pred-key", type=str, default="annotations", help="Key in the prediction JSON objects that contains the predicted spans.")
    return parser.parse_args()


def char_spans_to_word_mask(
    text: str,
    spans: list[tuple[int, int]],
):
    """
    spans: list of (start, end), end is exclusive

    Returns:
      words: list of words
      mask:  list of 0/1, same length as words
    """
    words_with_offsets = split_chunk_into_words(text)

    mask = []
    for _, w_start, w_end in words_with_offsets:
        in_span = 0
        for s_start, s_end in spans:
            # overlap check
            if not (w_end <= s_start or w_start >= s_end):
                in_span = 1
                break
        mask.append(in_span)

    words = [w for w, _, _ in words_with_offsets]
    return words, mask


def create_labels(items):
    all_labels = []
    for item in items:
        text = item["text"]
        spans = [(ann["start"], ann["end"]) for ann in item["annotations"]]
        _, mask = char_spans_to_word_mask(text, spans)
    
        all_labels.append(np.array(mask, dtype=int))

    return all_labels


def evaluate(predictions: list[dict], ground_truth: list[dict]) -> dict:
    all_gt_labels = create_labels(ground_truth)
    all_pred_labels = create_labels(predictions)

    all_TPS = []
    all_FPs = []
    all_FNs = []
    all_TNs = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_accuracies = []
    all_iou_scores = []

    all_binary_TPs = []
    all_binary_FPs = []
    all_binary_FNs = []
    all_binary_TNs = []

    for i, (gt_labels, pred_labels) in enumerate(zip(all_gt_labels, all_pred_labels)):
        TP = sum((gt_labels == 1) & (pred_labels == 1))
        FP = sum((gt_labels == 0) & (pred_labels == 1))
        FN = sum((gt_labels == 1) & (pred_labels == 0))
        TN = sum((gt_labels == 0) & (pred_labels == 0))
        all_TPS.append(TP)
        all_FPs.append(FP)
        all_FNs.append(FN)
        all_TNs.append(TN)

        # Calculate binary classification
        binary_pred = sum(pred_labels) > 0
        binary_gt = sum(gt_labels) > 0
        binary_TP = int(binary_pred == 1 and binary_gt == 1)
        binary_FP = int(binary_pred == 1 and binary_gt == 0)
        binary_FN = int(binary_pred == 0 and binary_gt == 1)
        binary_TN = int(binary_pred == 0 and binary_gt == 0)
        all_binary_TPs.append(binary_TP)
        all_binary_FPs.append(binary_FP)
        all_binary_FNs.append(binary_FN)
        all_binary_TNs.append(binary_TN)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_iou_scores.append(iou)
        all_accuracies.append(accuracy)

    total_TP = sum(all_TPS)
    total_FP = sum(all_FPs)
    total_FN = sum(all_FNs)
    total_TN = sum(all_TNs)

    micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    micro_accuracy = (total_TP + total_TN) / (total_TP + total_FP + total_FN + total_TN) if (total_TP + total_FP + total_FN + total_TN) > 0 else 0.0

    binary_total_TP = sum(all_binary_TPs)
    binary_total_FP = sum(all_binary_FPs)
    binary_total_FN = sum(all_binary_FNs)
    binary_total_TN = sum(all_binary_TNs)

    binary_precision = binary_total_TP / (binary_total_TP + binary_total_FP) if (binary_total_TP + binary_total_FP) > 0 else 0.0
    binary_recall = binary_total_TP / (binary_total_TP + binary_total_FN) if (binary_total_TP + binary_total_FN) > 0 else 0.0
    binary_f1 = 2 * binary_precision * binary_recall / (binary_precision + binary_recall) if (binary_precision + binary_recall) > 0 else 0.0

    # Acc@iou for iou thresholds 0.5, 0.75, 0.9
    iou_thresholds = [0.5, 0.75, 0.9]
    acc_at_iou = {}
    for threshold in iou_thresholds:
        acc_at_iou[f"acc@iou_{threshold}"] = sum(1 for iou in all_iou_scores if iou >= threshold) / len(all_iou_scores)

    result = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_accuracy": micro_accuracy,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
        "macro_precision": np.mean(all_precisions),
        "macro_recall": np.mean(all_recalls),
        "macro_f1": np.mean(all_f1s),
        "macro_accuracy": np.mean(all_accuracies),
        "macro_iou": np.mean(all_iou_scores),
        **acc_at_iou,
    }
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    args = parse_args()

    logger.info(f"Loading ground truth data from {args.gt} ...")
    with args.gt.open("r") as f:
        gt = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(gt)} samples from ground truth data.")

    logger.info(f"Loading predictions from {args.pred} ...")
    with args.pred.open("r") as f:
        pred = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(pred)} samples from predictions data.")

    assert len(pred) == len(gt), "Number of predictions and ground truth samples must be the same"

    if args.pred_key != "annotations":
        logger.info(f"Using '{args.pred_key}' as the key for predicted spans in the predictions data.")
        for item in pred:
            item["annotations"] = item.pop(args.pred_key)

    logger.info("Evaluating predictions against ground truth ...")
    results = evaluate(predictions=pred, ground_truth=gt)

    logger.info("Evaluation completed. Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
