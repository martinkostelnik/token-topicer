from pathlib import Path
import json
import argparse

import torch
import transformers

from token_topicer.cross_encoder.train import CrossEncoderTopicClassifierModule, cross_dot_product
from token_topicer.utils import prepare_sample_for_model, extract_spans


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

    def model_forward(
        self,
        text: str,
        topic_name: str,
        topic_description: str | None = None,
    ):
        result = {
            "text": text,
            "topic_name": topic_name,
            "topic_description": topic_description,
            "annotations": [],
            "probs": None,
        }

        sample = prepare_sample_for_model(
            text=text,
            topic=topic_name,
            tokenizer=self.tokenizer,
            include_topic_description=self.checkpoint["datamodule_hyper_parameters"]["include_topic_description"],
            topic_description=topic_description,
        )

        input_ids = sample["input_ids"].unsqueeze(0).to(self.device)
        token_type_ids = sample["token_type_ids"].unsqueeze(0).to(self.device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(self.device)

        sep_index = (sample["input_ids"] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()
        sample["token_type_ids"] = torch.cat([
            torch.zeros(sep_index + 1, dtype=torch.long),  # +1 to include SEP token in topic segment
            torch.ones(len(sample["input_ids"]) - (sep_index + 1), dtype=torch.long),
        ])
        token_type_ids = sample["token_type_ids"].unsqueeze(0).to(self.device)

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

        result["probs"] = probs
        result["offset_mapping"] = sample["offset_mapping"]

        return result
    
    def predict_from_probs(
        self,
        model_output: dict,
        threshold: float = 0.5,
    ) -> dict:
        model_output["annotations"] = []

        probs = model_output["probs"]
        preds = (probs >= threshold).long()
        
        spans = extract_spans(
            predictions=preds.squeeze(0).cpu().tolist(),
            offset_mapping=model_output["offset_mapping"],
        )

        for span in spans:
            model_output["annotations"].append({
                "start": span[0],
                "end": span[1],
                "text_piece": model_output["text"][span[0]:span[1]],
            })

        return model_output

    def predict(
        self,
        text: str,
        topic_name: str,
        topic_description: str | None = None,
        threshold: float | None = 0.5,
    ):
        result = self.model_forward(
            text=text,
            topic_name=topic_name,
            topic_description=topic_description,
        )
        
        if threshold is not None:
            result = self.predict_from_probs(
                model_output=result,
                threshold=threshold,
            )

        return result

    def predict_jsonl(
        self,
        input_data_path: Path,
        threshold: float | None = 0.5,
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

        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Cross-Encoder Topic Classifier")
    parser.add_argument("--model", type=Path, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Path to the input data for inference")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification")
    parser.add_argument("--save-path", type=Path, default=None, help="Path to save the predictions with the specified threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference_module = CrossEncoderInferenceModule(model_path=args.model)
    result = inference_module.predict_jsonl(input_data_path=args.data, threshold=args.threshold)

    if args.save_path is not None:
        with args.save_path.open("w")  as f:
            for item in result:
                item.pop("offset_mapping")  # Remove offset mapping from output for cleaner results
                item.pop("probs")  # Remove raw probabilities from output for cleaner results
                f.write(json.dumps(item) + "\n")