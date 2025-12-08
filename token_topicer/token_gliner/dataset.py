from pathlib import Path
import json
from typing import Any
import random
from dataclasses import dataclass, field
import logging

import lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)
    

def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_tokens = max(item["input_ids"].shape[0] for item in batch)
    max_entities = max(item["labels"].shape[1] for item in batch)
    min_sep_index = min(item["sep_token_index"] for item in batch)
    max_text_tokens = max_tokens - min_sep_index - 1
    # print("Max tokens in batch:", max_tokens)
    # print("Max entities in batch:", max_entities)
    # print("Min sep index in batch:", min_sep_index)
    # print("Max text tokens in batch:", max_text_tokens)

    input_ids = pad_sequence(
        [torch.cat([item["input_ids"], torch.zeros(max_tokens - item["input_ids"].shape[0], dtype=torch.long)]) for item in batch],
        batch_first=True,
    )

    attention_mask = pad_sequence(
        [torch.cat([item["attention_mask"], torch.zeros(max_tokens - item["attention_mask"].shape[0], dtype=torch.long)]) for item in batch],
        batch_first=True,
    )

    padded_labels = [torch.nn.functional.pad(sample["labels"], (0, max_entities - sample["labels"].shape[1]), value=-1) for sample in batch]
    padded_labels = pad_sequence(
        [torch.cat([lbl, torch.full((max_text_tokens - lbl.shape[0], max_entities), -1, dtype=torch.long)]) for lbl in padded_labels],
        batch_first=True,
        padding_value=-1,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": padded_labels,
        "max_entities": max_entities,
        "entity_token_indices": [sample["entity_token_indices"] for sample in batch],
        "sep_token_index": [sample["sep_token_index"] for sample in batch],
    }


@dataclass
class Entity:
    label: str
    id: int
    cluster_ids: list[int]
    input_ids: torch.Tensor = field(default_factory=torch.Tensor)


@dataclass
class TextSample:
    cluster_id: int
    input_ids: torch.Tensor
    entity_ids: list[int]
    labels: torch.Tensor


class TokenGlinerDataModule(L.LightningDataModule):
    def __init__(
        self,
        json_path_train: Path,
        json_path_val: Path,
        batch_size: int,
        tokenizer_path: Path,
    ) -> None:
        super().__init__()

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
        logger.info(f"Loading dataset from {json_path} ...")
        self.json_data = [item for item in load_json(json_path) if len(item["annotations"]) > 0]
        logger.info(f"Loaded {len(self.json_data)} samples from {json_path}.")

        self.tokenizer = tokenizer

        logger.info("Loading labels and entities ...")
        self.label2id: dict[str, int] = {} # Gets populated in load_labels
        self.label2entity: dict[str, Entity] = {} # Gets populated in load_labels
        self.entities: list[Entity] = self.load_labels()
        self.n_clusters = max([sample["cluster_id"] for sample in self.json_data]) + 1
        self.cluster_id2entity: dict[int, list[Entity]] = {i: [e for e in self.entities if i in e.cluster_ids] for i in range(self.n_clusters)}
        logger.info(f"Loaded {len(self.entities)} unique entities.")
        logger.info(f"First entity: {self.entities[0]}")

        self.ENT_TOKEN_ID: torch.Tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids("[ENT]")], dtype=torch.long)
        self.SEP_TOKEN_ID: torch.Tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])], dtype=torch.long)
        logger.info(f"Entity token ID: {self.ENT_TOKEN_ID}, SEP token ID: {self.SEP_TOKEN_ID}")

        logger.info("Pre-tokenizing dataset with multi-labels ...")
        self.data: list[TextSample] = self.pretokenize_with_multilabels()
        logger.info("Dataset preparation complete.")
        logger.info(f"First sample: {self.data[0]}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        idx = idx % len(self)
        
        text_sample: TextSample = self.data[idx]
        encoded_entities = torch.cat([torch.cat([self.ENT_TOKEN_ID, self.entities[eid].input_ids]) for eid in text_sample.entity_ids], dim=0)
        input_ids = torch.cat([encoded_entities, self.SEP_TOKEN_ID, text_sample.input_ids], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # Only take the corresponding labels
        labels = torch.index_select(
            input=text_sample.labels,
            dim=-1,
            index=torch.tensor(text_sample.entity_ids, dtype=torch.long)
        )

        if input_ids.shape[0] > 512:
            input_ids = input_ids[:512]
            attention_mask = attention_mask[:512]
            labels = labels[:512, :]

        ent_indices = torch.where(input_ids == self.ENT_TOKEN_ID)[0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "entity_token_indices": ent_indices,
            "sep_token_index": len(encoded_entities),
        }

    def load_labels(self) -> list[Entity]:
        entities = []
        processed_entities_set = set()
        for item in self.json_data:
            for ann in item["annotations"]:
                label = ann["label"]
                if label in processed_entities_set:
                    if item["cluster_id"] not in self.label2entity[label].cluster_ids:
                        self.label2entity[label].cluster_ids.append(item["cluster_id"])
                else:
                    processed_entities_set.add(label)
                    entity = Entity(
                        label=label,
                        id=len(entities),
                        cluster_ids=[item["cluster_id"]],
                        input_ids=self.tokenizer.encode(label, add_special_tokens=False, return_tensors="pt").squeeze(0),
                    )
                    self.label2id[label] = entity.id
                    self.label2entity[label] = entity
                    entities.append(entity)

        return entities

    def pretokenize_with_multilabels(self):
        result = []

        for item in self.json_data:
            text = item["text"]
            annotations = item["annotations"]

            enc = self.tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )

            offsets = enc["offset_mapping"]
            n_tokens = len(offsets)
            n_labels = len(self.entities)

            # multi-hot matrix: (n_tokens, n_labels)
            token_labels = torch.zeros((n_tokens, n_labels), dtype=torch.float32)

            for ann in annotations:
                label_id = self.label2id[ann["label"]]
                start, end = ann["start"], ann["end"]

                for tok_id, (tok_start, tok_end) in enumerate(offsets):
                    if tok_end > start and tok_start < end:
                        token_labels[tok_id, label_id] = 1.0

            text_sample = TextSample(
                cluster_id=item["cluster_id"],
                input_ids=torch.tensor(enc["input_ids"], dtype=torch.long),
                entity_ids=list(set([self.label2id[ann["label"]] for ann in annotations])),
                labels=token_labels,
            )
            result.append(text_sample)

        return result
    