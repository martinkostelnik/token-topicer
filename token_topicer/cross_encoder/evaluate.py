import argparse
import json
from pathlib import Path

from token_topicer.cross_encoder.inference import CrossEncoderInferenceModule
from token_topicer.utils import split_chunk_into_words

from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Cross Encoder model for topic tagging.")
    parser.add_argument("--pred", type=Path, required=True, help="Path to the predictions file.")
    parser.add_argument("--gt", type=Path, required=True, help="Path to the ground truth data file.")
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


def evaluate(predictions: list[dict], ground_truth: list[dict]) -> dict:
    all_pred_labels = []
    all_gt_labels = []

    for pred_item, gt_item in zip(predictions, ground_truth):
        assert pred_item["text"] == gt_item["text"], "Text mismatch between prediction and ground truth"

        text = pred_item["text"]
        pred_spans = [(ann["start"], ann["end"]) for ann in pred_item["annotations"]]
        gt_spans = [(ann["start"], ann["end"]) for ann in gt_item["annotations"]]
        pred_words, pred_mask = char_spans_to_word_mask(text, pred_spans)
        gt_words, gt_mask = char_spans_to_word_mask(text, gt_spans)
        
        assert pred_words == gt_words, "Word tokenization mismatch between prediction and ground truth"
        all_pred_labels.append(pred_mask)
        all_gt_labels.append(gt_mask)

    micro_f1 = f1_score(
        y_true=[label for gt in all_gt_labels for label in gt],
        y_pred=[label for pred in all_pred_labels for label in pred],
    )

    micro_precision = precision_score(
        y_true=[label for gt in all_gt_labels for label in gt],
        y_pred=[label for pred in all_pred_labels for label in pred],
    )

    micro_recall = recall_score(
        y_true=[label for gt in all_gt_labels for label in gt],
        y_pred=[label for pred in all_pred_labels for label in pred],
    )

    ious = []
    for gt_mask, pred_mask in zip(all_gt_labels, all_pred_labels):
        iou = jaccard_score(
            y_true=gt_mask,
            y_pred=pred_mask,
            zero_division=0,  # Handle case where both gt and pred are all zeros
        )
        ious.append(iou)
    jaccard = sum(ious) / len(ious)

    result = {
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "iou": jaccard,
    }
    return result


def main():
    args = parse_args()

    with args.gt.open("r") as f:
        gt = [json.loads(line) for line in f]

    with args.pred.open("r") as f:
        pred = [json.loads(line) for line in f]

    results = evaluate(predictions=pred, ground_truth=gt)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
