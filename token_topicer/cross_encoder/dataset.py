from pathlib import Path
import json
from typing import Any

from dataclasses import dataclass, field
import logging

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
        batch_size: int,
        tokenizer_path: Path,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.json_path_train = json_path_train
        self.json_path_val = json_path_val
        self.batch_size = batch_size

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

        self.train_dataset = TokenGlinerDataset(self.json_path_train, self.tokenizer)
        self.val_dataset = TokenGlinerDataset(self.json_path_val, self.tokenizer)

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
    ) -> None:
        super().__init__()
        self.json_data = [item for item in load_json(json_path) if len(item["annotations"]) > 0]
        self.tokenizer = tokenizer
        self.data = self.load_data()

    def load_data(self) -> list:
        data = []

        n_pos_labels = 0
        n_total_labels = 0

        for item in self.json_data:
            text = item["text"]
            annotations = item["annotations"]

            for annotation in annotations:
                label_name = annotation["label"]
                tokenizer_output = self.tokenizer(
                    label_name,
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    return_offsets_mapping=True,
                )
                offsets = tokenizer_output["offset_mapping"].squeeze(0).tolist()
                input_ids = tokenizer_output["input_ids"].squeeze(0)
                attention_mask = tokenizer_output["attention_mask"].squeeze(0)
                token_type_ids = tokenizer_output["token_type_ids"].squeeze(0)
                labels = torch.zeros(len(offsets), dtype=torch.long)

                text_token_indices = (token_type_ids == 1).nonzero(as_tuple=True)[0]

                labels = torch.zeros(len(text_token_indices) - 1, dtype=torch.float32)

                # Build the binary mask for text tokens only
                for idx, i in enumerate(text_token_indices):
                    start, end = offsets[i]
                    if start == end:
                        continue
                    if end > annotation["start"] and start < annotation["end"]:
                        labels[idx] = 1.0
                        n_pos_labels += 1
                    n_total_labels += 1

                sample = {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }

                data.append(sample)

        self.pos_weight = (n_total_labels - n_pos_labels) / n_pos_labels
        return data


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        idx = idx % len(self)
        return {**self.data[idx], "pos_weight": self.pos_weight}
