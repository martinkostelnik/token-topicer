from pathlib import Path
import json
import argparse

import torch
import transformers

from token_topicer.cross_encoder.train import CrossEncoderTopicClassifierModule, cross_dot_product
from token_topicer.utils import prepare_sample_for_model


class CrossEncoderInferenceModule:
    def __init__(
        self,
        model_path: Path,
    ) -> None:
        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint = torch.load(self.model_path)
        self.model = CrossEncoderTopicClassifierModule.load_from_checkpoint(self.model_path)
        self.model.eval()
        self.model = self.model.to(self.device)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.checkpoint["hyper_parameters"]["lm_path"])

    def predict(
        self,
        text: str,
        topic_name: str,
        topic_description: str | None = None,
        threshold: float = 0.5,
    ):
        result = {
            "text": text,
            "topic_name": topic_name,
            "topic_description": topic_description,
            "annotations": []
        }

        sample = prepare_sample_for_model(
            text=text,
            topic=topic_name,
            tokenizer=self.tokenizer,
            include_topic_description=self.checkpoint["datamodule_hyper_parameters"]["include_topic_description"],
            topic_description=topic_description,
        )

        if sample is None:
            print("AAA")

        input_ids = sample["input_ids"].unsqueeze(0).to(self.device)
        token_type_ids = sample["token_type_ids"].unsqueeze(0).to(self.device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            similarity_matrix = cross_dot_product(
                model_outputs=model_outputs,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

            # Max over topic tokens
            if self.model.soft_max_score:
                scores = torch.logsumexp(similarity_matrix, dim=1)  # shape (B, S_text)
            else:
                scores = similarity_matrix.max(dim=1).values  # shape (B, S_text)

            probs = torch.sigmoid(scores)
            preds = (probs >= threshold).long()

        spans = self.extract_spans(
            predictions=preds.squeeze(0).cpu().tolist(),
            offset_mapping=sample["offset_mapping"],
        )

        for span in spans:
            result["annotations"].append({
                "start": span[0],
                "end": span[1],
                "text_piece": text[span[0]:span[1]],
            })

        return result

    def predict_jsonl(
        self,
        input_data_path: Path,
        save_path: Path,
        threshold: float = 0.5,
    ):
        with input_data_path.open("r") as f:
            data = [json.loads(line) for line in f]

        result = []
    
        for item in data:
            text = item["text"]
            topic_name = item["topic_name"]
            topic_description = item["topic_description"]

            result_item = self.predict(
                text=text,
                topic_name=topic_name,
                topic_description=topic_description,
                threshold=threshold,
            )
            result.append(result_item)
        
        with save_path.open("w") as f:
            for item in result:
                print(json.dumps(item, ensure_ascii=False), file=f)

        return result

    def extract_spans(self, predictions: list[int], offset_mapping: list[tuple[int, int]], gap_tolerance: int=1):
        result = []
        start_char, end_char = None, None
        gap_count = 0

        for pred, (offset_start, offset_end) in zip(predictions, offset_mapping[-len(predictions)-1:-1], strict=True):
            if pred == 1:
                if start_char is None:
                    start_char = offset_start
                    end_char = offset_end
                    gap_count = 0
                else:
                    if gap_count > 0:
                        end_char = offset_end
                        gap_count = 0
                    else:
                        end_char = offset_end
            else:
                if start_char is not None:
                    gap_count += 1
                    if gap_count > gap_tolerance:
                        result.append((start_char, end_char))
                        start_char, end_char = None, None
                        gap_count = 0

        if start_char is not None:
            result.append((start_char, end_char))

        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Cross-Encoder Topic Classifier")
    parser.add_argument("--model", type=Path, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Path to the input data for inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference_module = CrossEncoderInferenceModule(model_path=args.model)
    inference_module.predict_jsonl(input_data_path=args.data, threshold=0.482, save_path=Path("inference_results.jsonl"))
