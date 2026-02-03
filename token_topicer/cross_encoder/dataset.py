from pathlib import Path
import json
from typing import Any

from dataclasses import dataclass, field

import lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers


def load_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)
    

def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_tokens = max(item["input_ids"].shape[0] for item in batch)
    max_text_tokens = max(sum(item["token_type_ids"]).item() - 1 for item in batch)

    input_ids = pad_sequence(
        [torch.cat([item["input_ids"], torch.zeros(max_tokens - item["input_ids"].shape[0], dtype=torch.long)]) for item in batch],
        batch_first=True,
    )

    attention_mask = pad_sequence(
        [torch.cat([item["attention_mask"], torch.zeros(max_tokens - item["attention_mask"].shape[0], dtype=torch.long)]) for item in batch],
        batch_first=True,
    )

    token_type_ids = pad_sequence(
        [torch.cat([item["token_type_ids"], torch.zeros(max_tokens - item["token_type_ids"].shape[0], dtype=torch.long)]) for item in batch],
        batch_first=True,
    )

    labels = pad_sequence(
        [torch.cat([item["labels"], -1 * torch.ones(max_text_tokens - item["labels"].shape[0], dtype=torch.long)]) for item in batch],
        batch_first=True,
        padding_value=-1,
    )

    pos_weight = torch.tensor(batch[0]["pos_weight"], dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
        "pos_weight": pos_weight,
    }


@dataclass
class Entity:
    label: str
    id: int
    cluster_ids: list[int]
    input_ids: torch.Tensor = field(default_factory=torch.Tensor)


class CrossEncoderTopicClassifierDataModule(L.LightningDataModule):
    def __init__(
        self,
        json_path_train: Path,
        json_path_val: Path,
        cluster_topics_path: Path,
        batch_size: int,
        tokenizer_path: Path,
        max_length: int = 512,
        include_topic_description: bool = False,
        data_limit: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.json_path_train = json_path_train
        self.json_path_val = json_path_val
        self.cluster_topics_path = cluster_topics_path
        self.include_topic_description = include_topic_description
        self.data_limit = data_limit
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

        self.train_dataset = TokenGlinerDataset(self.json_path_train, self.tokenizer, self.include_topic_description, self.max_length, self.data_limit)
        self.val_dataset = TokenGlinerDataset(self.json_path_val, self.tokenizer, self.include_topic_description, self.max_length, self.data_limit)
        
    def setup(self, stage: str | None = None) -> None:
        self.tokenizer.save_pretrained(self.trainer.default_root_dir)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )


class TokenGlinerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_path: Path,
        tokenizer: transformers.PreTrainedTokenizer,
        include_topic_description: bool = False,
        max_length: int = 512,
        data_limit: int | None = None,
        return_labels: bool = True,
    ) -> None:
        super().__init__()
        
        with open(json_path, "r") as f:
            self.jsonl_data = [json.loads(line) for line in f]
        
        self.tokenizer = tokenizer
        self.include_topic_description = include_topic_description
        self.max_length = max_length
        self.data_limit = data_limit
        self.return_labels = return_labels

        self.data = self.load_data()

    def load_data(self):
        data = []
        n_pos_labels = 0
        n_total_labels = 0
        loaded_samples = 0
        for item in self.jsonl_data:
            text = item["text"]
            topic_name = item["topic_name"]
            topic_description = item["topic_description"]

            topic_text = f"{topic_name}" if not self.include_topic_description else f"{topic_name} - {topic_description}"

            tokenizer_output = self.tokenizer(
                topic_text,
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
            offsets = tokenizer_output["offset_mapping"].squeeze(0).tolist()
            input_ids = tokenizer_output["input_ids"].squeeze(0)
            attention_mask = tokenizer_output["attention_mask"].squeeze(0)
            token_type_ids = tokenizer_output["token_type_ids"].squeeze(0)

            if len(input_ids) >= self.max_length:
                continue

            sample = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            }

            if self.return_labels:
                annotations = item["annotations"]
                text_token_indices = (token_type_ids == 1).nonzero(as_tuple=True)[0]
                labels = torch.zeros(len(text_token_indices) - 1, dtype=torch.float32) # -1 to exclude last SEP token

                # Build the binary mask for text tokens only
                for idx, i in enumerate(text_token_indices):
                    start, end = offsets[i]
                    if start == end:
                        continue
                    for annotation in annotations:
                        if end > annotation["start"] and start < annotation["end"]:
                            labels[idx] = 1.0
                            n_pos_labels += 1
                n_total_labels += len(labels)

                sample["labels"] = labels

            data.append(sample)
            loaded_samples += 1
            if self.data_limit is not None and loaded_samples >= self.data_limit:
                break

        self.pos_weight = ((n_total_labels - n_pos_labels) / n_pos_labels) if self.return_labels else 1.0
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {**self.data[idx], "pos_weight": self.pos_weight}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--cluster-topics-path", type=Path, required=True)
    parser.add_argument("--model")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    print("Building dataset...")
    dataset = TokenGlinerDataset(
        json_path=args.json_path,
        cluster_topics_path=args.cluster_topics_path,
        tokenizer=tokenizer,
        include_topic_description=True,
    )

    print(len(dataset))
    from tqdm import tqdm

    print("Calculating input lengths...")
    lengths = []
    for item in tqdm(dataset):
        lengths.append(len(item["input_ids"]))

    print("Plotting histogram...")
    import matplotlib.pyplot as plt

    plt.hist(lengths, bins=50)
    plt.xlabel("Sequence Length")
    plt.ylabel("Count")
    plt.title("Input Sequence Length Distribution")
    plt.savefig("input_length_distribution_topic_description.png")