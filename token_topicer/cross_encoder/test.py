import argparse
from pathlib import Path

import torch
import transformers
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from token_topicer.cross_encoder.dataset import TokenGlinerDatasetWide

def parse_args():
    parser = argparse.ArgumentParser(description="Test Cross Encoder model for topic tagging.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--soft_max_score",
        action="store_true",
        help="Whether to use soft maximum when selecting score for each token.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Whether to calibrate the model scores using a held-out dataset.",
    )
    return parser.parse_args()


def cross_dot_product(
    model_outputs: torch.Tensor,
    token_type_ids: torch.Tensor,
) -> torch.Tensor:
    topic_mask = (token_type_ids == 0)
    text_mask = (token_type_ids == 1)

    topic_tokens = model_outputs[topic_mask.bool()][1:-1]  # exclude CLS and SEP
    text_tokens = model_outputs[text_mask.bool()][:-1]     # exclude SEP

    similarity_matrix = torch.matmul(topic_tokens, text_tokens.T)  # shape (topic_len, text_len)
    similarity_matrix = similarity_matrix / torch.sqrt(torch.tensor(model_outputs.shape[-1], dtype=torch.float32))

    return similarity_matrix



def main():
    args = parse_args()

    model = torch.jit.load(str(args.model / "model_cuda.pt"), map_location="cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    print("Loading dataset...")
    dataset = TokenGlinerDatasetWide(
        json_path=args.data,
        tokenizer=tokenizer,
        include_topic_description=True,
        max_length=512,
    )

    all_labels = []
    all_scores = []

    print("Running evaluation...")
    for sample in tqdm(dataset):
        input_ids = sample["input_ids"].unsqueeze(0).to("cuda")
        attention_mask = sample["attention_mask"].unsqueeze(0).to("cuda")
        token_type_ids = sample["token_type_ids"].unsqueeze(0).to("cuda")
        labels = sample["labels"]
    
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        scores = cross_dot_product(
            model_outputs=outputs,
            token_type_ids=token_type_ids,
        )

        max_similarities = torch.max(scores, dim=0)[0] if not args.soft_max_score else torch.logsumexp(scores, dim=0)
        probs = torch.sigmoid(max_similarities).cpu().tolist()
        labels = labels.tolist()

        assert len(probs) == len(labels)
        all_scores.extend(probs)
        all_labels.extend(labels)

    print("Calculating metrics...")
    roc_auc = roc_auc_score(all_labels, all_scores)
    average_precision = average_precision_score(all_labels, all_scores)

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")

    if args.calibrate:
        print("Calibrating threshold...")
        best_threshold = -1
        best_precision = -1
        best_recall = -1
        best_f1 = -1
        for threshold in tqdm([i * 0.01 for i in range(1, 100)]):
            preds = [1 if score >= threshold else 0 for score in all_scores]
            precision = precision_score(all_labels, preds)
            recall = recall_score(all_labels, preds)
            f1 = f1_score(all_labels, preds)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall

    print(f"Threshold: {best_threshold:.2f} | F1: {best_f1:.4f} | Precision: {best_precision:.4f} | Recall: {best_recall:.4f}")


if __name__ == "__main__":
    main()